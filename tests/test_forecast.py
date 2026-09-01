"""Layer 5 tests for the rolling cash forecaster (src/forecast/cashflow.py).

Written before implementation per CLAUDE.md's build protocol. DB-backed
(session + batch_run_id), consistent with CLAUDE.md's rule against mocking
the ledger layer -- runs against the same real Postgres test database as
test_ledger.py (see tests/conftest.py).
"""
import json
import uuid
from datetime import date
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from src.common.money import to_paise
from src.data.models import BankStatementLine, GatewaySettlement, GroundTruthEntry, InternalOrder
from src.forecast.cashflow import (
    CONFIDENCE_BAND_PCT,
    RAIL_WINDOW_BUSINESS_DAYS,
    INTERNATIONAL_WINDOW_BUSINESS_DAYS,
    project_cashflow,
    rail_window_business_days,
)
from src.ledger.journal import (
    post_clean_match_settlement,
    post_honest_exception,
    post_order_capture,
    post_refund_clawback_reversal,
    post_utr_batch_settlement,
)
from src.matching.fast_path import run_fast_path
from src.matching.schema import bank_lines_to_frame, orders_to_frame, settlements_to_frame

FROZEN_DIR = Path(__file__).resolve().parents[1] / "data" / "challenge_batch_100"


# ---------------------------------------------------------------------------
# Fixtures: frozen dataset, parsed once per module (same pattern as test_ledger.py)
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
    for order_id in group.order_ids:
        s = settlements_by_order[order_id]
        post_order_capture(db_session, batch_run_id, order_id, to_paise(s.gross_amount))

    if len(group.order_ids) == 1:
        order_id = group.order_ids[0]
        fields = _settlement_paise_fields(settlements_by_order[order_id])
        entry = post_clean_match_settlement(
            db_session, batch_run_id, order_id,
            gross_paise=fields["gross_paise"], mdr_paise=fields["mdr_paise"],
            gst_paise=fields["gst_paise"], tds_paise=fields["tds_paise"], net_paise=fields["net_paise"],
        )
        return [entry]

    orders = [_settlement_paise_fields(settlements_by_order[oid]) for oid in group.order_ids]
    bank_credit_entry, settlement_entries = post_utr_batch_settlement(
        db_session, batch_run_id, group.utr, group.net_amount_paise, orders
    )
    return [bank_credit_entry] + settlement_entries


# ---------------------------------------------------------------------------
# Tests 1-3 -- rail_window_business_days
# ---------------------------------------------------------------------------

def test_rail_window_upi_is_t_plus_1():
    assert rail_window_business_days("upi", is_international=False) == 1


def test_rail_window_domestic_card_and_netbanking_is_t_plus_2():
    for rail in ("credit_card", "debit_card", "amex", "netbanking"):
        assert rail_window_business_days(rail, is_international=False) == 2


def test_rail_window_is_international_forces_t_plus_3_regardless_of_rail():
    for rail in ("credit_card", "amex", "netbanking"):
        assert rail_window_business_days(rail, is_international=True) == 3
    assert INTERNATIONAL_WINDOW_BUSINESS_DAYS == 3
    assert set(RAIL_WINDOW_BUSINESS_DAYS) == {"upi", "credit_card", "debit_card", "amex", "netbanking"}


# ---------------------------------------------------------------------------
# Test 4 -- business-day window, NOT raw calendar days. Expected date is a
# literal, hand-computed value against a real calendar (Friday 2025-01-10 +
# 1 business day = Monday 2025-01-13, skipping the weekend) -- not derived
# from any function in src/common/calendar.py, to avoid circularity with the
# code under test.
# ---------------------------------------------------------------------------

def test_window_uses_business_days_not_calendar_days(db_session, batch_run_id):
    order = InternalOrder(
        order_id="ORD_WEEKEND1",
        gross_amount=Decimal("1000.00"),
        customer_id="CUST_WEEKEND1",
        payment_method="upi",
        timestamp="2025-01-10T10:00:00+05:30",  # a Friday
    )
    settlement = GatewaySettlement(
        payment_id="PAY_WEEKEND1", order_id="ORD_WEEKEND1", gross_amount=Decimal("1000.00"),
        payment_method="upi", mdr=Decimal("0.00"), gst_on_mdr=Decimal("0.00"), tds_194o=Decimal("1.00"),
        net_amount=Decimal("999.00"), utr="UTR_WEEKEND1", settlement_date=date(2025, 1, 13),
    )
    post_order_capture(db_session, batch_run_id, order.order_id, to_paise(order.gross_amount))

    result = project_cashflow(
        db_session, batch_run_id, [order], [settlement], as_of=date(2025, 1, 10),
    )
    row = result.filter(pl.col("order_id") == "ORD_WEEKEND1")
    assert row["expected_cash_date"][0] == date(2025, 1, 13), (
        "hand-computed: Friday 2025-01-10 + 1 business day skips Sat/Sun, lands Monday "
        "2025-01-13 -- a naive calendar-day T+1 would wrongly give Saturday 2025-01-11"
    )


# ---------------------------------------------------------------------------
# Test 5 -- confirmed cash only counts orders with cash actually posted
# ---------------------------------------------------------------------------

def test_confirmed_cash_only_counts_orders_with_full_cash_posted(db_session, batch_run_id):
    order_confirmed = InternalOrder(
        order_id="ORD_CONF1", gross_amount=Decimal("500.00"), customer_id="C1",
        payment_method="upi", timestamp="2025-01-06T10:00:00+05:30",
    )
    settlement_confirmed = GatewaySettlement(
        payment_id="PAY_CONF1", order_id="ORD_CONF1", gross_amount=Decimal("500.00"),
        payment_method="upi", mdr=Decimal("0.00"), gst_on_mdr=Decimal("0.00"), tds_194o=Decimal("0.50"),
        net_amount=Decimal("499.50"), utr="UTR_CONF1", settlement_date=date(2025, 1, 7),
    )
    post_order_capture(db_session, batch_run_id, "ORD_CONF1", to_paise(order_confirmed.gross_amount))
    post_clean_match_settlement(
        db_session, batch_run_id, "ORD_CONF1",
        gross_paise=to_paise(settlement_confirmed.gross_amount), mdr_paise=0, gst_paise=0,
        tds_paise=50, net_paise=49_950,
    )

    order_pending = InternalOrder(
        order_id="ORD_PEND1", gross_amount=Decimal("700.00"), customer_id="C2",
        payment_method="upi", timestamp="2025-01-06T10:00:00+05:30",
    )
    settlement_pending = GatewaySettlement(
        payment_id="PAY_PEND1", order_id="ORD_PEND1", gross_amount=Decimal("700.00"),
        payment_method="upi", mdr=Decimal("0.00"), gst_on_mdr=Decimal("0.00"), tds_194o=Decimal("0.70"),
        net_amount=Decimal("699.30"), utr="UTR_PEND1", settlement_date=date(2025, 1, 7),
    )
    post_order_capture(db_session, batch_run_id, "ORD_PEND1", to_paise(order_pending.gross_amount))

    result = project_cashflow(
        db_session, batch_run_id, [order_confirmed, order_pending],
        [settlement_confirmed, settlement_pending], as_of=date(2025, 1, 6),
    )

    confirmed_row = result.filter(pl.col("order_id") == "ORD_CONF1")
    pending_row = result.filter(pl.col("order_id") == "ORD_PEND1")
    assert confirmed_row["account_status"][0] == "confirmed"
    assert confirmed_row["amount_paise"][0] == 49_950
    assert pending_row["account_status"][0] == "projected"
    assert pending_row["amount_paise"][0] == to_paise(settlement_pending.net_amount)


# ---------------------------------------------------------------------------
# Test 6 -- utr_batch order confirmed only once the sibling bank-credit
# entry has ALSO posted, not merely once its own Stage 2 entry exists. This
# is the case a naive "has Stage 2 posted" check would get wrong.
# ---------------------------------------------------------------------------

def test_utr_batch_order_confirmed_only_once_bank_credit_also_posted(db_session, batch_run_id):
    utr = "UTR_PARTIAL1"
    orders_paise = [
        {"order_id": "ORD_UTRP_A", "gross_paise": 100_000, "mdr_paise": 1000, "gst_paise": 180, "tds_paise": 100, "net_paise": 98_720},
        {"order_id": "ORD_UTRP_B", "gross_paise": 200_000, "mdr_paise": 2000, "gst_paise": 360, "tds_paise": 200, "net_paise": 197_440},
    ]
    for o in orders_paise:
        post_order_capture(db_session, batch_run_id, o["order_id"], o["gross_paise"])

    # Deliberately do NOT call post_utr_batch_settlement (which posts the
    # bank credit + both order legs together). Instead post only order A's
    # own Stage 2 leg directly via post_journal_entry-equivalent, to model a
    # state where Stage 2 posted for this order but the UTR's own bank
    # credit has not -- exercised via the ledger's real function but with
    # only a partial order set, so the sibling bank-credit entry never posts.
    from src.ledger.journal import (
        AR_GATEWAY_CLEARING, CASH_IN_TRANSIT_UTR, GST_ITC_RECEIVABLE, MDR_EXPENSE,
        TDS_194O_CREDIT, JournalEntrySpec, JournalLineSpec, post_journal_entry,
    )

    o = orders_paise[0]
    spec = JournalEntrySpec(
        batch_run_id=batch_run_id,
        idempotency_key=f"RUN:{batch_run_id}:ORDER:{o['order_id']}:UTR:{utr}:SETTLE",
        reference_id=o["order_id"],
        description="partial utr_batch settle, no bank credit yet",
        lines=[
            JournalLineSpec(account_code=CASH_IN_TRANSIT_UTR, direction="D", amount_paise=o["net_paise"]),
            JournalLineSpec(account_code=MDR_EXPENSE, direction="D", amount_paise=o["mdr_paise"]),
            JournalLineSpec(account_code=GST_ITC_RECEIVABLE, direction="D", amount_paise=o["gst_paise"]),
            JournalLineSpec(account_code=TDS_194O_CREDIT, direction="D", amount_paise=o["tds_paise"]),
            JournalLineSpec(account_code=AR_GATEWAY_CLEARING, direction="C", amount_paise=o["gross_paise"]),
        ],
    )
    post_journal_entry(db_session, spec)

    order = InternalOrder(
        order_id="ORD_UTRP_A", gross_amount=Decimal("1000.00"), customer_id="C1",
        payment_method="credit_card", timestamp="2025-01-06T10:00:00+05:30",
    )
    settlement = GatewaySettlement(
        payment_id="PAY_UTRP_A", order_id="ORD_UTRP_A", gross_amount=Decimal("1000.00"),
        payment_method="credit_card", mdr=Decimal("10.00"), gst_on_mdr=Decimal("1.80"), tds_194o=Decimal("1.00"),
        net_amount=Decimal("987.20"), utr=utr, settlement_date=date(2025, 1, 7),
    )

    result = project_cashflow(db_session, batch_run_id, [order], [settlement], as_of=date(2025, 1, 6))
    row = result.filter(pl.col("order_id") == "ORD_UTRP_A")
    assert row["account_status"][0] == "projected", (
        "Stage 2 posted but the sibling UTR bank-credit entry never landed -- "
        "must not be confirmed"
    )
    assert row["amount_paise"][0] == 98_720


# ---------------------------------------------------------------------------
# Test 7 -- projected amounts carry a flat +/-5% confidence band
# ---------------------------------------------------------------------------

def test_projected_amount_carries_five_percent_confidence_band(db_session, batch_run_id):
    order = InternalOrder(
        order_id="ORD_BAND1", gross_amount=Decimal("1000.00"), customer_id="C1",
        payment_method="upi", timestamp="2025-01-06T10:00:00+05:30",
    )
    settlement = GatewaySettlement(
        payment_id="PAY_BAND1", order_id="ORD_BAND1", gross_amount=Decimal("1000.00"),
        payment_method="upi", mdr=Decimal("0.00"), gst_on_mdr=Decimal("0.00"), tds_194o=Decimal("1.00"),
        net_amount=Decimal("999.00"), utr="UTR_BAND1", settlement_date=date(2025, 1, 7),
    )
    post_order_capture(db_session, batch_run_id, "ORD_BAND1", to_paise(order.gross_amount))

    result = project_cashflow(db_session, batch_run_id, [order], [settlement], as_of=date(2025, 1, 6))
    row = result.filter(pl.col("order_id") == "ORD_BAND1")
    amount = row["amount_paise"][0]
    assert amount == 99_900
    assert row["low_paise"][0] == round(amount * float(1 - CONFIDENCE_BAND_PCT))
    assert row["high_paise"][0] == round(amount * float(1 + CONFIDENCE_BAND_PCT))
    assert row["low_paise"][0] == 94_905
    assert row["high_paise"][0] == 104_895


# ---------------------------------------------------------------------------
# Test 8 -- confirmed cash has no confidence band (it's a fact, not a guess)
# ---------------------------------------------------------------------------

def test_confirmed_cash_has_no_confidence_band(db_session, batch_run_id):
    order = InternalOrder(
        order_id="ORD_NOBAND1", gross_amount=Decimal("500.00"), customer_id="C1",
        payment_method="upi", timestamp="2025-01-06T10:00:00+05:30",
    )
    settlement = GatewaySettlement(
        payment_id="PAY_NOBAND1", order_id="ORD_NOBAND1", gross_amount=Decimal("500.00"),
        payment_method="upi", mdr=Decimal("0.00"), gst_on_mdr=Decimal("0.00"), tds_194o=Decimal("0.50"),
        net_amount=Decimal("499.50"), utr="UTR_NOBAND1", settlement_date=date(2025, 1, 7),
    )
    post_order_capture(db_session, batch_run_id, "ORD_NOBAND1", to_paise(order.gross_amount))
    post_clean_match_settlement(
        db_session, batch_run_id, "ORD_NOBAND1",
        gross_paise=to_paise(settlement.gross_amount), mdr_paise=0, gst_paise=0, tds_paise=50, net_paise=49_950,
    )

    result = project_cashflow(db_session, batch_run_id, [order], [settlement], as_of=date(2025, 1, 6))
    row = result.filter(pl.col("order_id") == "ORD_NOBAND1")
    assert row["account_status"][0] == "confirmed"
    assert row["low_paise"][0] == row["amount_paise"][0] == row["high_paise"][0] == 49_950


# ---------------------------------------------------------------------------
# Test 9 -- confirmed and projected are separate rows/statuses, never merged
# ---------------------------------------------------------------------------

def test_confirmed_and_projected_are_separate_columns_not_merged(db_session, batch_run_id):
    order_a = InternalOrder(
        order_id="ORD_SEP_A", gross_amount=Decimal("300.00"), customer_id="C1",
        payment_method="upi", timestamp="2025-01-06T10:00:00+05:30",
    )
    settlement_a = GatewaySettlement(
        payment_id="PAY_SEP_A", order_id="ORD_SEP_A", gross_amount=Decimal("300.00"),
        payment_method="upi", mdr=Decimal("0.00"), gst_on_mdr=Decimal("0.00"), tds_194o=Decimal("0.30"),
        net_amount=Decimal("299.70"), utr="UTR_SEP_A", settlement_date=date(2025, 1, 7),
    )
    post_order_capture(db_session, batch_run_id, "ORD_SEP_A", to_paise(order_a.gross_amount))
    post_clean_match_settlement(
        db_session, batch_run_id, "ORD_SEP_A",
        gross_paise=to_paise(settlement_a.gross_amount), mdr_paise=0, gst_paise=0, tds_paise=30, net_paise=29_970,
    )

    order_b = InternalOrder(
        order_id="ORD_SEP_B", gross_amount=Decimal("400.00"), customer_id="C2",
        payment_method="upi", timestamp="2025-01-06T10:00:00+05:30",
    )
    settlement_b = GatewaySettlement(
        payment_id="PAY_SEP_B", order_id="ORD_SEP_B", gross_amount=Decimal("400.00"),
        payment_method="upi", mdr=Decimal("0.00"), gst_on_mdr=Decimal("0.00"), tds_194o=Decimal("0.40"),
        net_amount=Decimal("399.60"), utr="UTR_SEP_B", settlement_date=date(2025, 1, 7),
    )
    post_order_capture(db_session, batch_run_id, "ORD_SEP_B", to_paise(order_b.gross_amount))

    result = project_cashflow(
        db_session, batch_run_id, [order_a, order_b], [settlement_a, settlement_b], as_of=date(2025, 1, 6),
    )
    assert set(result["account_status"].unique().to_list()) == {"confirmed", "projected"}
    confirmed_total = result.filter(pl.col("account_status") == "confirmed")["amount_paise"].sum()
    projected_total = result.filter(pl.col("account_status") == "projected")["amount_paise"].sum()
    assert confirmed_total == 29_970
    assert projected_total == to_paise(settlement_b.net_amount)


# ---------------------------------------------------------------------------
# Test 10 -- 7 calendar-day horizon bucketing: a day-8 order is excluded
# from within_horizon but still present (and counted) in the frame.
# ---------------------------------------------------------------------------

def test_seven_calendar_day_horizon_bucketing(db_session, batch_run_id):
    as_of = date(2025, 1, 6)  # a Monday
    order = InternalOrder(
        order_id="ORD_HORIZON1", gross_amount=Decimal("1000.00"), customer_id="C1",
        payment_method="netbanking", timestamp="2025-01-06T10:00:00+05:30",
    )
    # netbanking, domestic -> T+2 business days from 2025-01-06 (Monday) = 2025-01-08 (Wednesday),
    # well within a 7-calendar-day horizon. To land on day 8+ we instead
    # capture the order 6 business days before as_of... simpler: directly
    # assert on within_horizon using an as_of chosen so the T+2 window
    # lands on day 9. 2025-01-06 (Mon) + 7 business days -> use amex intl (T+3)
    # from a date far enough that day+3 exceeds the horizon by construction
    # is awkward with small windows, so instead we shrink horizon_days.
    settlement = GatewaySettlement(
        payment_id="PAY_HORIZON1", order_id="ORD_HORIZON1", gross_amount=Decimal("1000.00"),
        payment_method="netbanking", mdr=Decimal("10.00"), gst_on_mdr=Decimal("1.80"), tds_194o=Decimal("1.00"),
        net_amount=Decimal("987.20"), utr="UTR_HORIZON1", settlement_date=date(2025, 1, 8),
    )
    post_order_capture(db_session, batch_run_id, "ORD_HORIZON1", to_paise(order.gross_amount))

    # horizon_days=1 forces the T+2 (2025-01-08) expected date outside the
    # 1-day horizon from 2025-01-06, while the order must still appear in
    # the returned frame and contribute to the unconditional total.
    result = project_cashflow(
        db_session, batch_run_id, [order], [settlement], as_of=as_of, horizon_days=1,
    )
    row = result.filter(pl.col("order_id") == "ORD_HORIZON1")
    assert len(row) == 1, "order must still appear in the frame even though its date is beyond the horizon"
    assert row["within_horizon"][0] is False
    assert row["amount_paise"][0] == to_paise(settlement.net_amount)
    assert result["amount_paise"].sum() == to_paise(settlement.net_amount), (
        "grand total (used for the sanity check) must include beyond-horizon amounts"
    )


# ---------------------------------------------------------------------------
# Test 11 -- honest_exception orders are excluded from the projection
# ---------------------------------------------------------------------------

def test_honest_exception_orders_excluded_from_projection(db_session, batch_run_id):
    order = InternalOrder(
        order_id="ORD_SUSPENSE1", gross_amount=Decimal("200.00"), customer_id="C1",
        payment_method="upi", timestamp="2025-01-06T10:00:00+05:30",
    )
    post_order_capture(db_session, batch_run_id, "ORD_SUSPENSE1", to_paise(order.gross_amount))
    post_honest_exception(
        db_session, batch_run_id, "ORD_SUSPENSE1", "AR_GATEWAY_CLEARING", "C", 20_000,
        note="unresolved, no matching settlement",
    )

    result = project_cashflow(db_session, batch_run_id, [order], [], as_of=date(2025, 1, 6))
    assert len(result.filter(pl.col("order_id") == "ORD_SUSPENSE1")) == 0


# ---------------------------------------------------------------------------
# Test 12 -- sanity check: confirmed + projected total == sum(net_amount_paise)
# over every settlement that actually reached Stage 2 in this batch_run_id
# ---------------------------------------------------------------------------

def test_sanity_check_total_matches_settled_net_amount_sum(db_session, batch_run_id):
    orders, settlements = [], []
    for i, (rail, net) in enumerate([("upi", "499.50"), ("credit_card", "499.50")]):
        oid = f"ORD_SANE{i}"
        order = InternalOrder(
            order_id=oid, gross_amount=Decimal("500.00"), customer_id=f"C{i}",
            payment_method=rail, timestamp="2025-01-06T10:00:00+05:30",
        )
        settlement = GatewaySettlement(
            payment_id=f"PAY_SANE{i}", order_id=oid, gross_amount=Decimal("500.00"),
            payment_method=rail, mdr=Decimal("0.00"), gst_on_mdr=Decimal("0.00"), tds_194o=Decimal("0.50"),
            net_amount=Decimal(net), utr=f"UTR_SANE{i}", settlement_date=date(2025, 1, 8),
        )
        orders.append(order)
        settlements.append(settlement)
        post_order_capture(db_session, batch_run_id, oid, to_paise(order.gross_amount))
        if i == 0:
            # confirm this one fully
            post_clean_match_settlement(
                db_session, batch_run_id, oid, gross_paise=50_000, mdr_paise=0,
                gst_paise=0, tds_paise=50, net_paise=to_paise(Decimal(net)),
            )
        # the other stays projected (Stage 1 only)

    result = project_cashflow(db_session, batch_run_id, orders, settlements, as_of=date(2025, 1, 6))
    settled_total = sum(to_paise(s.net_amount) for s in settlements)
    assert result["amount_paise"].sum() == settled_total


# ---------------------------------------------------------------------------
# Test 13 -- full frozen-batch end-to-end, including the 3 refund_clawback
# records (Layer 3 fix, post_refund_clawback_reversal)
# ---------------------------------------------------------------------------

def test_project_cashflow_on_frozen_batch_end_to_end(
    db_session, batch_run_id, fast_path_result, frozen_orders, frozen_settlements
):
    orders_by_id = {o.order_id: o for o in frozen_orders}
    settlements_by_order = {s.order_id: s for s in frozen_settlements}

    for group in fast_path_result.resolved:
        _post_resolved_group(db_session, batch_run_id, group, settlements_by_order)

    refund_order_ids = [o.order_id for o in frozen_orders if o.refund_amount is not None]
    assert len(refund_order_ids) == 3
    for order_id in refund_order_ids:
        s = settlements_by_order[order_id]
        post_order_capture(db_session, batch_run_id, order_id, to_paise(s.gross_amount))
        post_clean_match_settlement(
            db_session, batch_run_id, order_id,
            gross_paise=to_paise(s.gross_amount), mdr_paise=to_paise(s.mdr),
            gst_paise=to_paise(s.gst_on_mdr), tds_paise=to_paise(s.tds_194o), net_paise=to_paise(s.net_amount),
        )
        post_refund_clawback_reversal(
            db_session, batch_run_id, order_id, to_paise(orders_by_id[order_id].refund_amount)
        )

    posted_order_ids = set(fast_path_result.resolved_order_ids) | set(refund_order_ids)
    result = project_cashflow(
        db_session, batch_run_id, frozen_orders, frozen_settlements, as_of=date(2025, 1, 6),
    )

    settled_total = sum(to_paise(settlements_by_order[oid].net_amount) for oid in posted_order_ids)
    assert result["amount_paise"].sum() == settled_total

    for oid in refund_order_ids:
        row = result.filter(pl.col("order_id") == oid)
        assert row["account_status"][0] == "confirmed"
        assert row["amount_paise"][0] == to_paise(settlements_by_order[oid].net_amount)

    for row in result.iter_rows(named=True):
        settlement = settlements_by_order[row["order_id"]]
        order = orders_by_id[row["order_id"]]
        expected_window = rail_window_business_days(order.payment_method, settlement.is_international)
        assert rail_window_business_days(row["payment_method"], row["is_international"]) == expected_window


# ---------------------------------------------------------------------------
# Test 14 -- scope ceiling: no hazard-rate/probability-curve/scenario surface
# (a deliberately blunt check -- the real enforcement is not writing that
# code; this only proves the public module surface stayed small)
# ---------------------------------------------------------------------------

def test_no_hazard_or_scenario_model_surface():
    import src.forecast.cashflow as cashflow_module

    public_names = [n for n in dir(cashflow_module) if not n.startswith("_")]
    forbidden_substrings = ("hazard", "probability", "scenario", "monte_carlo", "simulation")
    for name in public_names:
        lowered = name.lower()
        for bad in forbidden_substrings:
            assert bad not in lowered, f"{name} suggests out-of-scope forecasting sophistication"
