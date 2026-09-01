"""Layer 5: rolling cash forecaster.

Projects a batch's in-flight funds into expected cash-available dates, using
rail-specific settlement windows in business days (src/common/calendar.py),
distinguishing cash already confirmed (posted to CASH) from cash still
projected (in-flight, per docs/plan.md Layer 5). This exists to prove the
reconciled ledger data is usable downstream, not to be a forecasting product
in its own right -- no hazard-rate model, no chargeback probability curves,
no scenario simulation.

Design (approved before implementation, see conversation record):
- DB-backed: reads posting state from Postgres (session + batch_run_id),
  consistent with CLAUDE.md's rule against mocking the ledger layer.
- Rail windows: UPI T+1; is_international=True forces T+3 regardless of
  rail; every other domestic rail (credit_card, debit_card, amex,
  netbanking) is T+2.
- Confidence band: a flat +/-5% on projected (not-yet-confirmed) amounts
  only. Confirmed amounts are facts, not projections, and carry no band.
- Horizon: an explicit `as_of` date (never date.today(), for determinism)
  plus 7 calendar days forward.

Per-order confirmed/projected status is read from the ledger, not
re-derived from the source records, mirroring the CASH_IN_TRANSIT_UTR
netting logic from Layer 3: a clean_match order's Stage 2 entry debits CASH
directly, so Stage 2 posted == confirmed. A utr_batch order's Stage 2 entry
only debits CASH_IN_TRANSIT_UTR -- it is confirmed only once the sibling
UTR bank-credit entry (which debits CASH) has also posted; otherwise it is
still projected, using its own already-known allocated cash share.

Orders whose only ledger trace is a SUSPENSE_UNRESOLVED entry (honest
exceptions -- including, currently, refund_clawback records that an
orchestration layer has not yet posted past Stage 1/2) have no
algorithmically-knowable cash date and are excluded from the projection
entirely, not given a fabricated one.
"""
from __future__ import annotations

import uuid
from datetime import date, timedelta
from decimal import ROUND_HALF_UP, Decimal

import polars as pl
from sqlalchemy import func as sa_func
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.common.calendar import add_business_days
from src.common.money import to_paise
from src.data.models import GatewaySettlement, InternalOrder
from src.ledger.journal import AR_GATEWAY_CLEARING, CASH, CASH_IN_TRANSIT_UTR, SUSPENSE_UNRESOLVED
from src.ledger.models import Account, JournalEntry, JournalLine

RAIL_WINDOW_BUSINESS_DAYS = {
    "upi": 1,
    "credit_card": 2,
    "debit_card": 2,
    "amex": 2,
    "netbanking": 2,
}
INTERNATIONAL_WINDOW_BUSINESS_DAYS = 3
CONFIDENCE_BAND_PCT = Decimal("0.05")


def rail_window_business_days(payment_method: str, is_international: bool) -> int:
    """UPI T+1, cards T+2, international/other T+3 -- is_international
    overrides the rail-specific window regardless of which rail it rode in
    on (docs/plan.md Layer 5, design decision confirmed before
    implementation)."""
    if is_international:
        return INTERNATIONAL_WINDOW_BUSINESS_DAYS
    return RAIL_WINDOW_BUSINESS_DAYS[payment_method]


def _confidence_band(amount_paise: int) -> tuple[int, int]:
    amt = Decimal(amount_paise)
    low = (amt * (Decimal("1") - CONFIDENCE_BAND_PCT)).to_integral_value(rounding=ROUND_HALF_UP)
    high = (amt * (Decimal("1") + CONFIDENCE_BAND_PCT)).to_integral_value(rounding=ROUND_HALF_UP)
    return int(low), int(high)


def _account_totals(session: Session, batch_run_id: uuid.UUID, reference_id: str) -> dict[tuple[str, str], int]:
    rows = session.execute(
        select(Account.account_code, JournalLine.direction, sa_func.coalesce(sa_func.sum(JournalLine.amount), 0))
        .join(JournalLine, JournalLine.account_id == Account.account_id)
        .join(JournalEntry, JournalEntry.entry_id == JournalLine.entry_id)
        .where(JournalEntry.batch_run_id == batch_run_id, JournalEntry.reference_id == reference_id)
        .group_by(Account.account_code, JournalLine.direction)
    ).all()
    return {(code, direction): to_paise(total) for code, direction, total in rows}


def _utr_bank_credit_posted(session: Session, batch_run_id: uuid.UUID, utr: str) -> bool:
    totals = _account_totals(session, batch_run_id, utr)
    return totals.get((CASH, "D"), 0) > 0


def project_cashflow(
    session: Session,
    batch_run_id: uuid.UUID,
    orders: list[InternalOrder],
    settlements: list[GatewaySettlement],
    as_of: date,
    horizon_days: int = 7,
) -> pl.DataFrame:
    """Returns one row per in-flight order (confirmed or projected).

    Columns: order_id, payment_method, is_international, account_status
    ("confirmed" | "projected"), expected_cash_date (None for confirmed),
    within_horizon (bool), amount_paise, low_paise, high_paise (== amount_paise
    for confirmed rows -- facts carry no band).

    Orders with no ledger trace at all in this batch_run_id, or whose only
    trace is SUSPENSE_UNRESOLVED, are omitted entirely. Callers wanting a
    day-by-day chart or a grand total both derive it from this one frame --
    filter on within_horizon for the former, sum amount_paise unconditionally
    for the latter (docs/plan.md Layer 5 "sanity-checked against the known
    batch total").
    """
    settlements_by_order = {s.order_id: s for s in settlements}
    horizon_end = as_of + timedelta(days=horizon_days)

    rows: list[dict] = []
    for order in orders:
        settlement = settlements_by_order.get(order.order_id)
        totals = _account_totals(session, batch_run_id, order.order_id)
        if not totals:
            continue  # not captured in this batch_run_id's ledger at all
        if totals.get((SUSPENSE_UNRESOLVED, "D"), 0) > 0 or totals.get((SUSPENSE_UNRESOLVED, "C"), 0) > 0:
            continue  # honest exception -- no algorithmically-knowable cash date

        ar_credited = totals.get((AR_GATEWAY_CLEARING, "C"), 0) > 0
        cash_debit = totals.get((CASH, "D"), 0)
        transit_debit = totals.get((CASH_IN_TRANSIT_UTR, "D"), 0)

        if ar_credited and cash_debit > 0:
            status, amount_paise, expected_date = "confirmed", cash_debit, None
        elif ar_credited and transit_debit > 0:
            if settlement is not None and _utr_bank_credit_posted(session, batch_run_id, settlement.utr):
                status, amount_paise, expected_date = "confirmed", transit_debit, None
            else:
                window = rail_window_business_days(order.payment_method, settlement.is_international if settlement else False)
                status = "projected"
                amount_paise = transit_debit
                expected_date = add_business_days(order.timestamp.date(), window)
        elif settlement is not None:
            # Stage 1 captured, Stage 2 not yet posted -- project using the
            # settlement's known net figure (the amount that will land once
            # it clears).
            window = rail_window_business_days(order.payment_method, settlement.is_international)
            status = "projected"
            amount_paise = to_paise(settlement.net_amount)
            expected_date = add_business_days(order.timestamp.date(), window)
        else:
            continue  # captured but no settlement known -- nothing to project

        if status == "confirmed":
            low_paise, high_paise = amount_paise, amount_paise
            within_horizon = True
        else:
            low_paise, high_paise = _confidence_band(amount_paise)
            within_horizon = as_of < expected_date <= horizon_end

        rows.append(
            {
                "order_id": order.order_id,
                "payment_method": order.payment_method,
                "is_international": bool(settlement.is_international) if settlement is not None else False,
                "account_status": status,
                "expected_cash_date": expected_date,
                "within_horizon": within_horizon,
                "amount_paise": amount_paise,
                "low_paise": low_paise,
                "high_paise": high_paise,
            }
        )

    return pl.DataFrame(
        rows,
        schema={
            "order_id": pl.Utf8,
            "payment_method": pl.Utf8,
            "is_international": pl.Boolean,
            "account_status": pl.Utf8,
            "expected_cash_date": pl.Date,
            "within_horizon": pl.Boolean,
            "amount_paise": pl.Int64,
            "low_paise": pl.Int64,
            "high_paise": pl.Int64,
        },
    )
