"""Layer 3: posting logic and gatekeeper.

Money is handled internally as integer paise (matching Layer 2's convention)
and converted to/from Decimal rupees only at the DB boundary, via
`src/common/money.py` (CLAUDE.md Sec.3). Every posting function goes through
`post_journal_entry`, which is the single place that (a) revalidates the
entry balances in application code via `JournalEntrySpec` before any DB
write -- the fast-fail, with the DB's deferred constraint trigger as the hard
backstop -- and (b) is idempotent on `idempotency_key`, scoped by
`batch_run_id` (CLAUDE.md Sec.5 / plan Sec.3.3): calling it twice with the
same key returns the existing entry rather than posting a duplicate.
"""
from __future__ import annotations

import uuid
from decimal import Decimal
from typing import Literal

import polars as pl
from pydantic import BaseModel, field_validator, model_validator
from sqlalchemy import func as sa_func
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.common.money import from_paise, to_paise
from src.ledger.allocation import allocate_utr_batch, assert_allocation_gap_within_cap
from src.ledger.models import Account, JournalEntry, JournalLine

Direction = Literal["D", "C"]

CASH = "CASH"
CASH_IN_TRANSIT_UTR = "CASH_IN_TRANSIT_UTR"
AR_GATEWAY_CLEARING = "AR_GATEWAY_CLEARING"
REVENUE_GROSS = "REVENUE_GROSS"
MDR_EXPENSE = "MDR_EXPENSE"
GST_ITC_RECEIVABLE = "GST_ITC_RECEIVABLE"
TDS_194O_CREDIT = "TDS_194O_CREDIT"
ROUNDING_DIFFERENCE = "ROUNDING_DIFFERENCE"
SUSPENSE_UNRESOLVED = "SUSPENSE_UNRESOLVED"


class JournalLineSpec(BaseModel):
    account_code: str
    direction: Direction
    amount_paise: int

    @field_validator("amount_paise")
    @classmethod
    def positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("amount_paise must be strictly positive")
        return v


class JournalEntrySpec(BaseModel):
    """The application-layer fast-fail: revalidates debits == credits before
    any DB write is attempted. The DB's deferred constraint trigger is the
    hard backstop if this is ever bypassed."""

    batch_run_id: uuid.UUID
    idempotency_key: str
    reference_id: str
    description: str = ""
    lines: list[JournalLineSpec]

    @model_validator(mode="after")
    def balanced(self) -> "JournalEntrySpec":
        if len(self.lines) < 2:
            raise ValueError("a journal entry needs at least 2 lines")
        debit_total = sum(l.amount_paise for l in self.lines if l.direction == "D")
        credit_total = sum(l.amount_paise for l in self.lines if l.direction == "C")
        if debit_total != credit_total:
            raise ValueError(
                f"unbalanced journal entry: debits {debit_total} paise != credits {credit_total} paise"
            )
        return self


def post_journal_entry(session: Session, spec: JournalEntrySpec) -> JournalEntry:
    existing = session.execute(
        select(JournalEntry).where(JournalEntry.idempotency_key == spec.idempotency_key)
    ).scalar_one_or_none()
    if existing is not None:
        return existing

    codes = {line.account_code for line in spec.lines}
    account_ids = dict(
        session.execute(
            select(Account.account_code, Account.account_id).where(Account.account_code.in_(codes))
        ).all()
    )
    missing = codes - account_ids.keys()
    if missing:
        raise ValueError(f"unknown account code(s): {sorted(missing)}")

    entry = JournalEntry(
        batch_run_id=spec.batch_run_id,
        idempotency_key=spec.idempotency_key,
        reference_id=spec.reference_id,
        description=spec.description,
    )
    session.add(entry)
    session.flush()

    for line in spec.lines:
        session.add(
            JournalLine(
                entry_id=entry.entry_id,
                account_id=account_ids[line.account_code],
                direction=line.direction,
                amount=from_paise(line.amount_paise),
            )
        )
    session.commit()
    return entry


def post_order_capture(
    session: Session, batch_run_id: uuid.UUID, order_id: str, gross_amount_paise: int
) -> JournalEntry:
    """Stage 1, at order capture (T0): Debit AR_GATEWAY_CLEARING, Credit REVENUE_GROSS."""
    spec = JournalEntrySpec(
        batch_run_id=batch_run_id,
        idempotency_key=f"RUN:{batch_run_id}:ORDER:{order_id}:CAPTURE",
        reference_id=order_id,
        description=f"Order capture: {order_id}",
        lines=[
            JournalLineSpec(account_code=AR_GATEWAY_CLEARING, direction="D", amount_paise=gross_amount_paise),
            JournalLineSpec(account_code=REVENUE_GROSS, direction="C", amount_paise=gross_amount_paise),
        ],
    )
    return post_journal_entry(session, spec)


def post_clean_match_settlement(
    session: Session,
    batch_run_id: uuid.UUID,
    order_id: str,
    gross_paise: int,
    mdr_paise: int,
    gst_paise: int,
    tds_paise: int,
    net_paise: int,
) -> JournalEntry:
    """Non-batched clean_match: Stage 2 and the UTR bank credit collapse into
    one entry, debiting CASH directly and crediting AR_GATEWAY_CLEARING."""
    lines = [JournalLineSpec(account_code=CASH, direction="D", amount_paise=net_paise)]
    if mdr_paise:
        lines.append(JournalLineSpec(account_code=MDR_EXPENSE, direction="D", amount_paise=mdr_paise))
    if gst_paise:
        lines.append(JournalLineSpec(account_code=GST_ITC_RECEIVABLE, direction="D", amount_paise=gst_paise))
    if tds_paise:
        lines.append(JournalLineSpec(account_code=TDS_194O_CREDIT, direction="D", amount_paise=tds_paise))
    lines.append(JournalLineSpec(account_code=AR_GATEWAY_CLEARING, direction="C", amount_paise=gross_paise))

    spec = JournalEntrySpec(
        batch_run_id=batch_run_id,
        idempotency_key=f"RUN:{batch_run_id}:ORDER:{order_id}:SETTLE",
        reference_id=order_id,
        description=f"Clean-match settlement: {order_id}",
        lines=lines,
    )
    return post_journal_entry(session, spec)


def post_utr_batch_settlement(
    session: Session,
    batch_run_id: uuid.UUID,
    utr: str,
    total_credit_paise: int,
    orders: list[dict],
) -> tuple[JournalEntry, list[JournalEntry]]:
    """Posts the UTR lump-sum bank credit once, then a per-order Stage 2
    entry for each sibling order.

    `orders` is a list of dicts with keys order_id, gross_paise, mdr_paise,
    gst_paise, tds_paise, net_paise (each order's own factual settlement
    figures). `total_credit_paise` is split across orders via
    largest-remainder allocation (src/ledger/allocation.py), weighted by each
    order's own net_paise. Any per-order gap between its allocated cash share
    and its true net_paise -- zero whenever the shares already sum to
    total_credit_paise, and never more than a few paise in aggregate
    otherwise -- is posted to ROUNDING_DIFFERENCE, never fudged into
    MDR/GST/TDS or the gross amount, which stay factual (CLAUDE.md Sec.6).
    """
    shares = [(o["order_id"], o["net_paise"]) for o in orders]
    assert_allocation_gap_within_cap(total_credit_paise, shares)
    allocated = allocate_utr_batch(total_credit_paise, shares)

    bank_credit_entry = post_journal_entry(
        session,
        JournalEntrySpec(
            batch_run_id=batch_run_id,
            idempotency_key=f"RUN:{batch_run_id}:UTR:{utr}:BANK_CREDIT",
            reference_id=utr,
            description=f"UTR lump-sum bank credit: {utr}",
            lines=[
                JournalLineSpec(account_code=CASH, direction="D", amount_paise=total_credit_paise),
                JournalLineSpec(
                    account_code=CASH_IN_TRANSIT_UTR, direction="C", amount_paise=total_credit_paise
                ),
            ],
        ),
    )

    settlement_entries: list[JournalEntry] = []
    for o in orders:
        allocated_paise = allocated[o["order_id"]]
        signed_residual = allocated_paise - o["net_paise"]

        lines = [
            JournalLineSpec(account_code=CASH_IN_TRANSIT_UTR, direction="D", amount_paise=allocated_paise)
        ]
        if o["mdr_paise"]:
            lines.append(JournalLineSpec(account_code=MDR_EXPENSE, direction="D", amount_paise=o["mdr_paise"]))
        if o["gst_paise"]:
            lines.append(
                JournalLineSpec(account_code=GST_ITC_RECEIVABLE, direction="D", amount_paise=o["gst_paise"])
            )
        if o["tds_paise"]:
            lines.append(JournalLineSpec(account_code=TDS_194O_CREDIT, direction="D", amount_paise=o["tds_paise"]))
        if signed_residual < 0:
            lines.append(
                JournalLineSpec(
                    account_code=ROUNDING_DIFFERENCE, direction="D", amount_paise=-signed_residual
                )
            )
        lines.append(JournalLineSpec(account_code=AR_GATEWAY_CLEARING, direction="C", amount_paise=o["gross_paise"]))
        if signed_residual > 0:
            lines.append(
                JournalLineSpec(account_code=ROUNDING_DIFFERENCE, direction="C", amount_paise=signed_residual)
            )

        entry = post_journal_entry(
            session,
            JournalEntrySpec(
                batch_run_id=batch_run_id,
                idempotency_key=f"RUN:{batch_run_id}:ORDER:{o['order_id']}:UTR:{utr}:SETTLE",
                reference_id=o["order_id"],
                description=f"UTR-batch settlement: {o['order_id']} (UTR {utr})",
                lines=lines,
            ),
        )
        settlement_entries.append(entry)

    return bank_credit_entry, settlement_entries


def post_refund_clawback_reversal(
    session: Session, batch_run_id: uuid.UUID, order_id: str, refund_amount_paise: int
) -> JournalEntry:
    """Refund revenue-recognition reversal (plan Sec.1.4 refund_clawback /
    CLAUDE.md Sec.1's generator addendum): Debit REVENUE_GROSS, Credit
    AR_GATEWAY_CLEARING for the refund amount. MDR_EXPENSE,
    GST_ITC_RECEIVABLE and TDS_194O_CREDIT are never touched here -- the
    merchant does not recover the processing fee on a refunded transaction,
    which is the entire point of the refund_clawback category.

    Note on AR_GATEWAY_CLEARING going net-negative for this order: Stage 2
    already credits AR_GATEWAY_CLEARING by the full gross amount, clearing
    the receivable to zero once the gateway has paid. This reversal credits
    it a second time, by the refund amount, so the account nets negative for
    the order. That is the intended mechanism, not a bug: it is the
    balancing residual for cash the merchant already received from the
    gateway but owes back to the customer directly -- a real-world payout
    this project deliberately does not model as a further CASH movement
    (out of scope, consistent with CLAUDE.md's discipline of stating a
    modeling assumption in one sentence rather than dodging it).
    """
    spec = JournalEntrySpec(
        batch_run_id=batch_run_id,
        idempotency_key=f"RUN:{batch_run_id}:ORDER:{order_id}:REFUND_REVERSAL",
        reference_id=order_id,
        description=f"Refund revenue reversal (MDR not reversed): {order_id}",
        lines=[
            JournalLineSpec(account_code=REVENUE_GROSS, direction="D", amount_paise=refund_amount_paise),
            JournalLineSpec(account_code=AR_GATEWAY_CLEARING, direction="C", amount_paise=refund_amount_paise),
        ],
    )
    return post_journal_entry(session, spec)


def post_honest_exception(
    session: Session,
    batch_run_id: uuid.UUID,
    reference_id: str,
    contra_account_code: str,
    contra_direction: Direction,
    amount_paise: int,
    note: str = "",
) -> JournalEntry:
    """Posts a balancing entry against SUSPENSE_UNRESOLVED so the books stay
    balanced even for a record no fast-path or agent resolution could
    explain (plan Sec.3.3)."""
    suspense_direction: Direction = "C" if contra_direction == "D" else "D"
    spec = JournalEntrySpec(
        batch_run_id=batch_run_id,
        idempotency_key=f"RUN:{batch_run_id}:BANK_TXN:{reference_id}:SUSPENSE",
        reference_id=reference_id,
        description=note or f"Honest exception: {reference_id}",
        lines=[
            JournalLineSpec(account_code=contra_account_code, direction=contra_direction, amount_paise=amount_paise),
            JournalLineSpec(account_code=SUSPENSE_UNRESOLVED, direction=suspense_direction, amount_paise=amount_paise),
        ],
    )
    return post_journal_entry(session, spec)


def assert_all_entries_have_balanced_lines(session: Session, batch_run_id: uuid.UUID | None = None) -> None:
    """Application-layer invariant: every journal_entries row has >= 2
    journal_lines rows, and they balance. A zero-line entry never fires the
    row-level trigger, so this closes that gap (db_schema.sql's documented
    residual)."""
    query = (
        select(
            JournalEntry.entry_id,
            sa_func.count(JournalLine.line_id),
            sa_func.coalesce(sa_func.sum(JournalLine.amount).filter(JournalLine.direction == "D"), 0),
            sa_func.coalesce(sa_func.sum(JournalLine.amount).filter(JournalLine.direction == "C"), 0),
        )
        .join(JournalLine, JournalLine.entry_id == JournalEntry.entry_id)
        .group_by(JournalEntry.entry_id)
    )
    if batch_run_id is not None:
        query = query.where(JournalEntry.batch_run_id == batch_run_id)

    for entry_id, line_count, debit_total, credit_total in session.execute(query).all():
        if line_count < 2:
            raise AssertionError(f"journal_entries.entry_id={entry_id} has fewer than 2 journal_lines rows")
        if debit_total != credit_total:
            raise AssertionError(
                f"journal_entries.entry_id={entry_id} unbalanced: debits={debit_total} credits={credit_total}"
            )


def trial_balance(session: Session, batch_run_id: uuid.UUID) -> pl.DataFrame:
    """Every account's closing debit/credit balance for this run, in integer
    paise, plus a final TOTAL row that must sum to zero."""
    rows = session.execute(
        select(
            Account.account_code,
            Account.account_name,
            Account.account_type,
            JournalLine.direction,
            sa_func.coalesce(sa_func.sum(JournalLine.amount), 0),
        )
        .join(JournalLine, JournalLine.account_id == Account.account_id)
        .join(JournalEntry, JournalEntry.entry_id == JournalLine.entry_id)
        .where(JournalEntry.batch_run_id == batch_run_id)
        .group_by(Account.account_code, Account.account_name, Account.account_type, JournalLine.direction)
    ).all()

    per_account: dict[str, dict[str, int]] = {}
    names: dict[str, str] = {}
    types: dict[str, str] = {}
    for account_code, account_name, account_type, direction, total in rows:
        names[account_code] = account_name
        types[account_code] = account_type
        per_account.setdefault(account_code, {"D": 0, "C": 0})
        per_account[account_code][direction] = to_paise(Decimal(total))

    record_rows = []
    for account_code in sorted(per_account):
        debit = per_account[account_code]["D"]
        credit = per_account[account_code]["C"]
        record_rows.append(
            {
                "account_code": account_code,
                "account_name": names[account_code],
                "account_type": types[account_code],
                "debit_total_paise": debit,
                "credit_total_paise": credit,
                "net_balance_paise": debit - credit,
            }
        )

    record_rows.append(
        {
            "account_code": "TOTAL",
            "account_name": "",
            "account_type": "",
            "debit_total_paise": sum(r["debit_total_paise"] for r in record_rows),
            "credit_total_paise": sum(r["credit_total_paise"] for r in record_rows),
            "net_balance_paise": sum(r["net_balance_paise"] for r in record_rows),
        }
    )
    return pl.DataFrame(record_rows)
