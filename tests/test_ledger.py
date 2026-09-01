"""Layer 3 tests for the PostgreSQL double-entry ledger
(src/ledger/db_schema.sql, models.py, allocation.py, journal.py).

Written before implementation per CLAUDE.md's build protocol. Runs against a
real Postgres test database (see tests/conftest.py) -- CLAUDE.md forbids
mocking the ENUM types / deferred constraint trigger this layer depends on.
"""
import json
import uuid
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError
from sqlalchemy import func as sa_func
from sqlalchemy import select, text
from sqlalchemy.exc import DBAPIError

from src.common.money import to_paise
from src.data.models import BankStatementLine, GatewaySettlement, GroundTruthEntry, InternalOrder
from src.ledger.allocation import (
    AllocationResidualTooLarge,
    allocate_utr_batch,
    assert_allocation_gap_within_cap,
)
from src.ledger.journal import (
    AR_GATEWAY_CLEARING,
    CASH,
    CASH_IN_TRANSIT_UTR,
    GST_ITC_RECEIVABLE,
    MDR_EXPENSE,
    REVENUE_GROSS,
    ROUNDING_DIFFERENCE,
    SUSPENSE_UNRESOLVED,
    TDS_194O_CREDIT,
    JournalEntrySpec,
    JournalLineSpec,
    assert_all_entries_have_balanced_lines,
    post_clean_match_settlement,
    post_honest_exception,
    post_order_capture,
    post_refund_clawback_reversal,
    post_utr_batch_settlement,
    trial_balance,
)
from src.ledger.models import Account, JournalEntry, JournalLine
from src.matching.fast_path import run_fast_path
from src.matching.schema import bank_lines_to_frame, orders_to_frame, settlements_to_frame

FROZEN_DIR = Path(__file__).resolve().parents[1] / "data" / "challenge_batch_100"


# ---------------------------------------------------------------------------
# Fixtures: the frozen challenge_batch_100 dataset, parsed once per module.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def frozen_orders() -> list[InternalOrder]:
    data = json.loads((FROZEN_DIR / "internal_orders.json").read_text(encoding="utf-8"))
    return [InternalOrder.model_validate(d) for d in data]


@pytest.fixture(scope="module")
def frozen_settlements() -> list[GatewaySettlement]:
    data = json.loads((FROZEN_DIR / "gateway_settlement.json").read_text(encoding="utf-8"))
    return [GatewaySettlement.model_validate(d) for d in data]


@pytest.fixture(scope="module")
def frozen_bank_lines() -> list[BankStatementLine]:
    data = json.loads((FROZEN_DIR / "bank_statement.json").read_text(encoding="utf-8"))
    return [BankStatementLine.model_validate(d) for d in data]


@pytest.fixture(scope="module")
def frozen_ground_truth() -> list[GroundTruthEntry]:
    data = json.loads((FROZEN_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    return [GroundTruthEntry.model_validate(d) for d in data]


@pytest.fixture(scope="module")
def fast_path_result(frozen_orders, frozen_settlements, frozen_bank_lines):
    orders_df = orders_to_frame(frozen_orders)
    settlements_df = settlements_to_frame(frozen_settlements)
    bank_df = bank_lines_to_frame(frozen_bank_lines)
    return run_fast_path(orders_df, settlements_df, bank_df)


def _settlement_paise_fields(s: GatewaySettlement) -> dict:
    return {
        "order_id": s.order_id,
        "gross_paise": to_paise(s.gross_amount),
        "mdr_paise": to_paise(s.mdr),
        "gst_paise": to_paise(s.gst_on_mdr),
        "tds_paise": to_paise(s.tds_194o),
        "net_paise": to_paise(s.net_amount),
    }


def _post_resolved_group(db_session, batch_run_id, group, settlements_by_order):
    """Posts Stage 1 (order capture) for every order in the group, then
    Stage 2 -- collapsed with the bank credit for a single-order clean_match,
    or via the UTR lump-sum + per-order allocation path for a multi-order
    utr_batch group."""
    for order_id in group.order_ids:
        s = settlements_by_order[order_id]
        post_order_capture(db_session, batch_run_id, order_id, to_paise(s.gross_amount))

    if len(group.order_ids) == 1:
        order_id = group.order_ids[0]
        fields = _settlement_paise_fields(settlements_by_order[order_id])
        entry = post_clean_match_settlement(
            db_session,
            batch_run_id,
            order_id,
            gross_paise=fields["gross_paise"],
            mdr_paise=fields["mdr_paise"],
            gst_paise=fields["gst_paise"],
            tds_paise=fields["tds_paise"],
            net_paise=fields["net_paise"],
        )
        return [entry]

    orders = [_settlement_paise_fields(settlements_by_order[oid]) for oid in group.order_ids]
    bank_credit_entry, settlement_entries = post_utr_batch_settlement(
        db_session, batch_run_id, group.utr, group.net_amount_paise, orders
    )
    return [bank_credit_entry] + settlement_entries


# ---------------------------------------------------------------------------
# Test 1 -- Pydantic pre-check rejects an unbalanced entry
# ---------------------------------------------------------------------------

def test_pydantic_rejects_unbalanced_entry(batch_run_id):
    with pytest.raises(ValidationError):
        JournalEntrySpec(
            batch_run_id=batch_run_id,
            idempotency_key=f"RUN:{batch_run_id}:TEST:UNBALANCED",
            reference_id="TEST_UNBALANCED",
            lines=[
                JournalLineSpec(account_code=CASH, direction="D", amount_paise=1000),
                JournalLineSpec(account_code=REVENUE_GROSS, direction="C", amount_paise=900),
            ],
        )


# ---------------------------------------------------------------------------
# Test 2 -- DB trigger is the hard backstop if the Pydantic check is bypassed
# ---------------------------------------------------------------------------

def test_db_trigger_rejects_unbalanced_entry_raw_sql(db_session, batch_run_id):
    cash_id = db_session.execute(select(Account.account_id).where(Account.account_code == CASH)).scalar_one()
    revenue_id = db_session.execute(
        select(Account.account_id).where(Account.account_code == REVENUE_GROSS)
    ).scalar_one()

    entry_id = db_session.execute(
        text(
            "INSERT INTO journal_entries (batch_run_id, idempotency_key, reference_id) "
            "VALUES (:batch_run_id, :key, :ref) RETURNING entry_id"
        ),
        {
            "batch_run_id": str(batch_run_id),
            "key": f"RUN:{batch_run_id}:TEST:RAWSQL_UNBALANCED",
            "ref": "TEST_RAWSQL",
        },
    ).scalar_one()
    db_session.execute(
        text("INSERT INTO journal_lines (entry_id, account_id, direction, amount) VALUES (:e, :a, 'D', :amt)"),
        {"e": entry_id, "a": cash_id, "amt": "10.00"},
    )
    db_session.execute(
        text("INSERT INTO journal_lines (entry_id, account_id, direction, amount) VALUES (:e, :a, 'C', :amt)"),
        {"e": entry_id, "a": revenue_id, "amt": "9.00"},
    )

    with pytest.raises(DBAPIError):
        db_session.commit()
    db_session.rollback()


# ---------------------------------------------------------------------------
# Test 3 -- deleting a line from a previously-balanced entry is caught
# ---------------------------------------------------------------------------

def test_db_trigger_rejects_deleted_line_leaving_imbalance(db_session, batch_run_id):
    entry = post_order_capture(db_session, batch_run_id, "ORD_DELTEST", 100_000)
    line_id = db_session.execute(
        select(JournalLine.line_id).where(JournalLine.entry_id == entry.entry_id).limit(1)
    ).scalar_one()

    db_session.execute(text("DELETE FROM journal_lines WHERE line_id = :lid"), {"lid": line_id})
    with pytest.raises(DBAPIError):
        db_session.commit()
    db_session.rollback()


# ---------------------------------------------------------------------------
# Test 4 -- every journal_entries row has >= 2 balanced journal_lines rows
# ---------------------------------------------------------------------------

def test_every_entry_has_at_least_two_balanced_lines(db_session, batch_run_id):
    post_order_capture(db_session, batch_run_id, "ORD_INV1", 50_000)
    post_clean_match_settlement(
        db_session,
        batch_run_id,
        "ORD_INV1",
        gross_paise=50_000,
        mdr_paise=500,
        gst_paise=90,
        tds_paise=50,
        net_paise=49_360,
    )
    assert_all_entries_have_balanced_lines(db_session, batch_run_id)


# ---------------------------------------------------------------------------
# Test 5 -- posting the same idempotency key twice does not duplicate
# ---------------------------------------------------------------------------

def test_posting_same_idempotency_key_twice_is_noop(db_session, batch_run_id):
    entry1 = post_order_capture(db_session, batch_run_id, "ORD_IDEMP1", 75_000)
    entry2 = post_order_capture(db_session, batch_run_id, "ORD_IDEMP1", 75_000)
    assert entry1.entry_id == entry2.entry_id

    count = db_session.execute(
        select(sa_func.count()).select_from(JournalEntry).where(
            JournalEntry.idempotency_key == entry1.idempotency_key
        )
    ).scalar_one()
    assert count == 1


# ---------------------------------------------------------------------------
# Test 6 -- GST and TDS post to asset accounts, never a liability
# ---------------------------------------------------------------------------

def test_gst_and_tds_post_to_asset_accounts(db_session, batch_run_id):
    post_order_capture(db_session, batch_run_id, "ORD_ASSET1", 100_000)
    entry = post_clean_match_settlement(
        db_session,
        batch_run_id,
        "ORD_ASSET1",
        gross_paise=100_000,
        mdr_paise=1000,
        gst_paise=180,
        tds_paise=100,
        net_paise=98_720,
    )
    rows = db_session.execute(
        select(Account.account_code, Account.account_type)
        .join(JournalLine, JournalLine.account_id == Account.account_id)
        .where(JournalLine.entry_id == entry.entry_id)
    ).all()
    type_by_code = dict(rows)
    assert type_by_code[GST_ITC_RECEIVABLE] == "asset"
    assert type_by_code[TDS_194O_CREDIT] == "asset"


# ---------------------------------------------------------------------------
# Test 7 -- REVENUE_GROSS is actually posted to at Stage 1
# ---------------------------------------------------------------------------

def test_revenue_gross_credited_at_capture(db_session, batch_run_id):
    entry = post_order_capture(db_session, batch_run_id, "ORD_REV1", 42_000)
    rows = db_session.execute(
        select(Account.account_code, JournalLine.direction, JournalLine.amount)
        .join(JournalLine, JournalLine.account_id == Account.account_id)
        .where(JournalLine.entry_id == entry.entry_id)
    ).all()
    revenue_rows = [(code, d, amt) for code, d, amt in rows if code == REVENUE_GROSS]
    assert len(revenue_rows) == 1
    _, direction, amount = revenue_rows[0]
    assert direction == "C"
    assert to_paise(amount) == 42_000


# ---------------------------------------------------------------------------
# Test 8 -- CASH_IN_TRANSIT_UTR nets to zero once a UTR batch fully settles
# ---------------------------------------------------------------------------

def test_cash_in_transit_nets_to_zero_after_utr_batch_settles(db_session, batch_run_id):
    utr = "UTRSYN0000001"
    orders = [
        {"order_id": "ORD_UTRA", "gross_paise": 100_000, "mdr_paise": 1000, "gst_paise": 180, "tds_paise": 100, "net_paise": 98_720},
        {"order_id": "ORD_UTRB", "gross_paise": 200_000, "mdr_paise": 2000, "gst_paise": 360, "tds_paise": 200, "net_paise": 197_440},
    ]
    for o in orders:
        post_order_capture(db_session, batch_run_id, o["order_id"], o["gross_paise"])

    total_credit_paise = sum(o["net_paise"] for o in orders)
    post_utr_batch_settlement(db_session, batch_run_id, utr, total_credit_paise, orders)

    tb = trial_balance(db_session, batch_run_id)
    row = tb.filter(pl.col("account_code") == CASH_IN_TRANSIT_UTR)
    assert row["net_balance_paise"][0] == 0


# ---------------------------------------------------------------------------
# Tests 9-11 -- allocate_utr_batch unit tests
# ---------------------------------------------------------------------------

def test_allocate_utr_batch_sums_exactly_no_residual_case():
    shares = [("A", 500), ("B", 300), ("C", 200)]
    allocated = allocate_utr_batch(1000, shares)
    assert allocated == {"A": 500, "B": 300, "C": 200}


def test_allocate_utr_batch_distributes_remainder_by_largest_fraction():
    shares = [("A", 1), ("B", 1), ("C", 1)]
    allocated = allocate_utr_batch(100, shares)
    assert sum(allocated.values()) == 100
    assert allocated == {"A": 34, "B": 33, "C": 33}


def test_allocate_utr_batch_residual_capped_and_routed_to_rounding_difference():
    # shares here model each order's own known net_paise (the real ledger
    # usage in post_utr_batch_settlement) -- sums to 100, so a bank credit
    # of 200 is a 100-paise gap across 3 shares, far exceeding the +/-3 cap.
    shares = [("A", 40), ("B", 30), ("C", 30)]
    with pytest.raises(AllocationResidualTooLarge):
        assert_allocation_gap_within_cap(200, shares)


# ---------------------------------------------------------------------------
# Test 12 -- every utr_batch record in the frozen dataset allocates exactly
# ---------------------------------------------------------------------------

def test_all_utr_batch_records_in_frozen_dataset_allocate_exactly(
    frozen_settlements, frozen_bank_lines, frozen_ground_truth
):
    batch_order_ids = {e.order_id for e in frozen_ground_truth if e.category == "utr_batch"}
    settlements_by_order = {s.order_id: s for s in frozen_settlements}
    bank_by_utr = {b.utr: b for b in frozen_bank_lines}

    groups: dict[str, list[GatewaySettlement]] = {}
    for oid in batch_order_ids:
        s = settlements_by_order[oid]
        groups.setdefault(s.utr, []).append(s)

    assert len(groups) >= 2, "expected multiple distinct utr_batch groups in the frozen dataset"

    for utr, members in groups.items():
        shares = [(s.order_id, to_paise(s.net_amount)) for s in members]
        total_paise = to_paise(bank_by_utr[utr].credited_amount)
        assert_allocation_gap_within_cap(total_paise, shares)
        allocated = allocate_utr_batch(total_paise, shares)
        assert sum(allocated.values()) == total_paise
        for order_id, share_paise in shares:
            assert allocated[order_id] == share_paise, (
                f"unexpected residual for {order_id}: allocated {allocated[order_id]} vs share {share_paise}"
            )


# ---------------------------------------------------------------------------
# Test 13 -- synthetic utr_batch WITH a genuine residual, full posting
# pipeline, proving ROUNDING_DIFFERENCE is exercised end-to-end (not just
# the standalone allocator).
# ---------------------------------------------------------------------------

def test_synthetic_utr_batch_with_residual_posts_rounding_difference_end_to_end(db_session, batch_run_id):
    utr = "UTRSYN_RESIDUAL_1"
    orders = [
        {"order_id": "ORD_RND_A", "gross_paise": 346, "mdr_paise": 10, "gst_paise": 2, "tds_paise": 1, "net_paise": 333},
        {"order_id": "ORD_RND_B", "gross_paise": 346, "mdr_paise": 10, "gst_paise": 2, "tds_paise": 1, "net_paise": 333},
        {"order_id": "ORD_RND_C", "gross_paise": 347, "mdr_paise": 10, "gst_paise": 2, "tds_paise": 1, "net_paise": 334},
    ]
    for o in orders:
        assert o["net_paise"] + o["mdr_paise"] + o["gst_paise"] + o["tds_paise"] == o["gross_paise"]
        post_order_capture(db_session, batch_run_id, o["order_id"], o["gross_paise"])

    total_credit_paise = 999  # 1 paise short of sum(net_paise) == 1000 -- a genuine, small, real gap
    assert sum(o["net_paise"] for o in orders) - total_credit_paise == 1

    bank_credit_entry, settlement_entries = post_utr_batch_settlement(
        db_session, batch_run_id, utr, total_credit_paise, orders
    )
    assert bank_credit_entry is not None
    assert len(settlement_entries) == 3

    rounding_lines = db_session.execute(
        select(JournalEntry.reference_id, JournalLine.direction, JournalLine.amount)
        .join(JournalLine, JournalLine.entry_id == JournalEntry.entry_id)
        .join(Account, Account.account_id == JournalLine.account_id)
        .where(Account.account_code == ROUNDING_DIFFERENCE, JournalEntry.batch_run_id == batch_run_id)
    ).all()

    assert len(rounding_lines) == 1, "exactly one order should absorb the 1-paise batch-level gap"
    reference_id, direction, amount = rounding_lines[0]
    assert reference_id == "ORD_RND_C"
    assert direction == "D"
    assert to_paise(amount) == 1

    assert_all_entries_have_balanced_lines(db_session, batch_run_id)

    tb = trial_balance(db_session, batch_run_id)
    cash_in_transit_row = tb.filter(pl.col("account_code") == CASH_IN_TRANSIT_UTR)
    assert cash_in_transit_row["net_balance_paise"][0] == 0


# ---------------------------------------------------------------------------
# Test 14 -- two batch_run_ids coexist with independent trial balances
# ---------------------------------------------------------------------------

def test_two_batch_runs_coexist_with_independent_trial_balances(db_session):
    run_a = uuid.uuid4()
    run_b = uuid.uuid4()

    post_order_capture(db_session, run_a, "ORD_RUNA_1", 10_000)
    post_clean_match_settlement(
        db_session, run_a, "ORD_RUNA_1",
        gross_paise=10_000, mdr_paise=100, gst_paise=18, tds_paise=10, net_paise=9_872,
    )

    post_order_capture(db_session, run_b, "ORD_RUNB_1", 20_000)
    post_clean_match_settlement(
        db_session, run_b, "ORD_RUNB_1",
        gross_paise=20_000, mdr_paise=200, gst_paise=36, tds_paise=20, net_paise=19_744,
    )

    tb_a = trial_balance(db_session, run_a)
    tb_b = trial_balance(db_session, run_b)

    assert tb_a.filter(pl.col("account_code") == "TOTAL")["net_balance_paise"][0] == 0
    assert tb_b.filter(pl.col("account_code") == "TOTAL")["net_balance_paise"][0] == 0

    assert tb_a.filter(pl.col("account_code") == CASH)["net_balance_paise"][0] == 9_872
    assert tb_b.filter(pl.col("account_code") == CASH)["net_balance_paise"][0] == 19_744


# ---------------------------------------------------------------------------
# Test 15 -- trial_balance() sums to exactly zero for a fully-settled run
# ---------------------------------------------------------------------------

def test_trial_balance_sums_to_zero_for_fully_settled_run(
    db_session, batch_run_id, fast_path_result, frozen_settlements
):
    settlements_by_order = {s.order_id: s for s in frozen_settlements}
    for group in fast_path_result.resolved:
        _post_resolved_group(db_session, batch_run_id, group, settlements_by_order)

    tb = trial_balance(db_session, batch_run_id)
    total_row = tb.filter(pl.col("account_code") == "TOTAL")
    assert total_row["net_balance_paise"][0] == 0


# ---------------------------------------------------------------------------
# Test 16 -- Layer 2's fast-path results post correctly into the ledger
# ---------------------------------------------------------------------------

def test_fast_path_results_post_correctly_into_ledger(
    db_session, batch_run_id, fast_path_result, frozen_settlements
):
    settlements_by_order = {s.order_id: s for s in frozen_settlements}
    for group in fast_path_result.resolved:
        _post_resolved_group(db_session, batch_run_id, group, settlements_by_order)

    assert_all_entries_have_balanced_lines(db_session, batch_run_id)

    for order_id in fast_path_result.resolved_order_ids:
        capture_key = f"RUN:{batch_run_id}:ORDER:{order_id}:CAPTURE"
        exists = db_session.execute(
            select(JournalEntry.entry_id).where(JournalEntry.idempotency_key == capture_key)
        ).scalar_one_or_none()
        assert exists is not None, order_id


# ---------------------------------------------------------------------------
# Test 17 -- honest_exception posts a balancing entry to SUSPENSE_UNRESOLVED
# ---------------------------------------------------------------------------

def test_honest_exception_posts_to_suspense(db_session, batch_run_id):
    entry = post_honest_exception(
        db_session,
        batch_run_id,
        "BANK_TXN_ORPHAN_1",
        CASH,
        "D",
        12_345,
        note="orphan bank credit, no matching settlement",
    )
    rows = db_session.execute(
        select(Account.account_code, JournalLine.direction)
        .join(JournalLine, JournalLine.account_id == Account.account_id)
        .where(JournalLine.entry_id == entry.entry_id)
    ).all()
    codes = {code for code, _ in rows}
    assert SUSPENSE_UNRESOLVED in codes
    assert_all_entries_have_balanced_lines(db_session, batch_run_id)


# ---------------------------------------------------------------------------
# Tests 18-21 -- Layer 3 fix found during Layer 5 design: refund_clawback
# orders had NO ledger posting path at all (fast_path correctly excludes
# them, and nothing wired an agent resolution to a ledger write).
# post_refund_clawback_reversal closes that gap: Debit REVENUE_GROSS, Credit
# AR_GATEWAY_CLEARING, MDR/GST/TDS untouched (plan Sec.1.4 refund_clawback).
# ---------------------------------------------------------------------------

def _post_refund_clawback_settlement(db_session, batch_run_id, order, settlement):
    """Posts the normal Stage1+Stage2 lifecycle for a refund_clawback order
    -- its settlement is computed on the original, unrefunded gross, so it
    is structurally identical to a clean_match order -- then the refund
    reversal entry on top."""
    gross_paise = to_paise(settlement.gross_amount)
    post_order_capture(db_session, batch_run_id, order.order_id, gross_paise)
    settlement_entry = post_clean_match_settlement(
        db_session,
        batch_run_id,
        order.order_id,
        gross_paise=gross_paise,
        mdr_paise=to_paise(settlement.mdr),
        gst_paise=to_paise(settlement.gst_on_mdr),
        tds_paise=to_paise(settlement.tds_194o),
        net_paise=to_paise(settlement.net_amount),
    )
    reversal_entry = post_refund_clawback_reversal(
        db_session, batch_run_id, order.order_id, to_paise(order.refund_amount)
    )
    return settlement_entry, reversal_entry


def test_refund_reversal_debits_revenue_credits_ar_gateway_clearing(db_session, batch_run_id):
    entry = post_refund_clawback_reversal(db_session, batch_run_id, "ORD_REFUND1", 55_552)
    rows = db_session.execute(
        select(Account.account_code, JournalLine.direction, JournalLine.amount)
        .join(JournalLine, JournalLine.account_id == Account.account_id)
        .where(JournalLine.entry_id == entry.entry_id)
    ).all()
    by_code = {code: (direction, to_paise(amount)) for code, direction, amount in rows}
    assert by_code[REVENUE_GROSS] == ("D", 55_552)
    assert by_code[AR_GATEWAY_CLEARING] == ("C", 55_552)
    assert_all_entries_have_balanced_lines(db_session, batch_run_id)


def test_refund_reversal_never_touches_mdr_expense(db_session, batch_run_id, frozen_orders, frozen_settlements):
    """Same rigor as the Layer 1 generator test and the Layer 4 agent test
    for this category: uses the frozen dataset's real MDR figures and proves
    the posted MDR_EXPENSE total equals them exactly, unreduced by the
    refund. Rs 15.80 / 83.04 / 17.87 MDR against Rs 555.52 / 1517.29 / 535.59
    refunds have no arithmetic relationship a netting bug could accidentally
    satisfy -- this cannot pass by coincidence."""
    orders_by_id = {o.order_id: o for o in frozen_orders}
    settlements_by_id = {s.order_id: s for s in frozen_settlements}
    refund_order_ids = [o.order_id for o in frozen_orders if o.refund_amount is not None]
    assert len(refund_order_ids) == 3

    for order_id in refund_order_ids:
        order = orders_by_id[order_id]
        settlement = settlements_by_id[order_id]
        _post_refund_clawback_settlement(db_session, batch_run_id, order, settlement)

        mdr_total = db_session.execute(
            select(sa_func.coalesce(sa_func.sum(JournalLine.amount), 0))
            .join(JournalEntry, JournalEntry.entry_id == JournalLine.entry_id)
            .join(Account, Account.account_id == JournalLine.account_id)
            .where(
                JournalEntry.batch_run_id == batch_run_id,
                JournalEntry.reference_id == order_id,
                Account.account_code == MDR_EXPENSE,
            )
        ).scalar_one()
        assert to_paise(mdr_total) == to_paise(settlement.mdr), order_id
        assert to_paise(mdr_total) > 0, order_id  # there must be something real to catch


def test_refund_reversal_idempotent(db_session, batch_run_id):
    entry1 = post_refund_clawback_reversal(db_session, batch_run_id, "ORD_REFUND_IDEMP", 10_000)
    entry2 = post_refund_clawback_reversal(db_session, batch_run_id, "ORD_REFUND_IDEMP", 10_000)
    assert entry1.entry_id == entry2.entry_id

    count = db_session.execute(
        select(sa_func.count()).select_from(JournalEntry).where(
            JournalEntry.idempotency_key == entry1.idempotency_key
        )
    ).scalar_one()
    assert count == 1


def test_refund_reversal_posted_for_all_frozen_refund_clawback_orders_trial_balance_still_balances(
    db_session, batch_run_id, frozen_orders, frozen_settlements
):
    orders_by_id = {o.order_id: o for o in frozen_orders}
    settlements_by_id = {s.order_id: s for s in frozen_settlements}
    refund_order_ids = [o.order_id for o in frozen_orders if o.refund_amount is not None]
    assert len(refund_order_ids) == 3

    for order_id in refund_order_ids:
        order = orders_by_id[order_id]
        settlement = settlements_by_id[order_id]
        _post_refund_clawback_settlement(db_session, batch_run_id, order, settlement)

    tb_after = trial_balance(db_session, batch_run_id)
    total_row = tb_after.filter(pl.col("account_code") == "TOTAL")
    assert total_row["net_balance_paise"][0] == 0

    revenue_row = tb_after.filter(pl.col("account_code") == REVENUE_GROSS)
    ar_row = tb_after.filter(pl.col("account_code") == AR_GATEWAY_CLEARING)

    gross_total = sum(to_paise(settlements_by_id[oid].gross_amount) for oid in refund_order_ids)
    refund_total = sum(to_paise(orders_by_id[oid].refund_amount) for oid in refund_order_ids)

    # REVENUE_GROSS: credited gross_total at Stage 1, debited refund_total by the reversal.
    assert revenue_row["credit_total_paise"][0] == gross_total
    assert revenue_row["debit_total_paise"][0] == refund_total
    assert revenue_row["net_balance_paise"][0] == refund_total - gross_total

    # AR_GATEWAY_CLEARING: debited gross_total at Stage 1, credited gross_total at
    # Stage 2 (clearing to zero), then credited refund_total again by the
    # reversal -- net negative, the documented intended residual.
    assert ar_row["debit_total_paise"][0] == gross_total
    assert ar_row["credit_total_paise"][0] == gross_total + refund_total
    assert ar_row["net_balance_paise"][0] == -refund_total
