"""Layer 2 tests for the fast-path matching cascade
(src/matching/schema.py, src/matching/fast_path.py). Written before
implementation per CLAUDE.md's build protocol.

Two design decisions, confirmed with the user before writing these tests,
are load-bearing for several of them (see fast_path.py's module docstring
for the full rationale):

1. The fast path revalidates expected MDR/GST(18%)/TDS(0.1%) and the
   expected settlement window, not just order<->payment<->UTR<->bank
   linking -- otherwise cutoff_drift/fee_drift/missing_tax_line/
   refund_clawback would incorrectly auto-resolve, since settlement and bank
   always tie out exactly for them (Layer 1 Addendum A1).
2. Hop 3 recovers the UTR from `narration` only -- the structured
   `bank_line.utr` column is never read as a matching key.
"""
import json
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from src.common.calendar import business_days_between
from src.common.money import from_paise, to_paise
from src.data.generator import TDS_RATE, _compute_settlement_fields
from src.data.models import BankStatementLine, GatewaySettlement, GroundTruthEntry, InternalOrder
from src.matching.fast_path import (
    compute_expected_rate_frame,
    expected_mdr_gst_tds_paise,
    run_fast_path,
)
from src.matching.schema import bank_lines_to_frame, orders_to_frame, settlements_to_frame

FROZEN_DIR = Path(__file__).resolve().parents[1] / "data" / "challenge_batch_100"

MONEY_FIELDS = {
    "internal_orders.json": ["gross_amount", "refund_amount"],
    "gateway_settlement.json": ["gross_amount", "mdr", "gst_on_mdr", "tds_194o", "net_amount"],
    "bank_statement.json": ["credited_amount"],
}


# ---------------------------------------------------------------------------
# Fixtures: the frozen challenge_batch_100 dataset, parsed and framed once.
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


def _order_ids_for_category(ground_truth: list[GroundTruthEntry], category: str) -> list[str]:
    return [e.order_id for e in ground_truth if e.category == category]


def _reason_by_order_id(result) -> dict[str, str]:
    return {oid: d.reason for d in result.discrepancies for oid in d.order_ids}


# ---------------------------------------------------------------------------
# Test 1 -- paise round-trip, full frozen batch, zero drift
# ---------------------------------------------------------------------------

def test_paise_roundtrip_full_batch():
    checked = 0
    for filename, fields in MONEY_FIELDS.items():
        data = json.loads((FROZEN_DIR / filename).read_text(encoding="utf-8"))
        for record in data:
            for field_name in fields:
                value = record.get(field_name)
                if value is None:
                    continue
                original = Decimal(value)
                assert from_paise(to_paise(original)) == original
                checked += 1
    assert checked > 300, "test is vacuous -- too few money fields were actually checked"


# ---------------------------------------------------------------------------
# Test 2/3 -- fast path resolves exactly clean_match+utr_batch, 60-70% range
# ---------------------------------------------------------------------------

def test_fast_path_resolves_expected_categories_only(fast_path_result, frozen_ground_truth):
    expected = {
        e.order_id for e in frozen_ground_truth if e.category in ("clean_match", "utr_batch")
    }
    assert fast_path_result.resolved_order_ids == expected


def test_fast_path_resolution_count_in_range(fast_path_result):
    count = len(fast_path_result.resolved_order_ids)
    assert 60 <= count <= 70, f"fast path resolved {count}/100, expected roughly 60-70"


# ---------------------------------------------------------------------------
# Tests 4-7 -- categories that tie out arithmetically but must still be
# excluded via rate/timing/refund revalidation (Layer 1 Addendum A1)
# ---------------------------------------------------------------------------

def test_cutoff_drift_excluded_from_fast_path(fast_path_result, frozen_ground_truth):
    reason_by_order = _reason_by_order_id(fast_path_result)
    ids = _order_ids_for_category(frozen_ground_truth, "cutoff_drift")
    assert len(ids) >= 1
    for oid in ids:
        assert oid not in fast_path_result.resolved_order_ids, oid
        assert reason_by_order.get(oid) == "rate_or_timing_or_refund_deviation", oid


def test_fee_drift_excluded_from_fast_path(fast_path_result, frozen_ground_truth):
    reason_by_order = _reason_by_order_id(fast_path_result)
    ids = _order_ids_for_category(frozen_ground_truth, "fee_drift")
    assert len(ids) >= 1
    for oid in ids:
        assert oid not in fast_path_result.resolved_order_ids, oid
        assert reason_by_order.get(oid) == "rate_or_timing_or_refund_deviation", oid


def test_missing_tax_line_excluded_from_fast_path(fast_path_result, frozen_ground_truth):
    reason_by_order = _reason_by_order_id(fast_path_result)
    ids = _order_ids_for_category(frozen_ground_truth, "missing_tax_line")
    assert len(ids) >= 1
    for oid in ids:
        assert oid not in fast_path_result.resolved_order_ids, oid
        assert reason_by_order.get(oid) == "rate_or_timing_or_refund_deviation", oid


def test_refund_clawback_excluded_from_fast_path(fast_path_result, frozen_ground_truth):
    reason_by_order = _reason_by_order_id(fast_path_result)
    ids = _order_ids_for_category(frozen_ground_truth, "refund_clawback")
    assert len(ids) >= 1
    for oid in ids:
        assert oid not in fast_path_result.resolved_order_ids, oid
        assert reason_by_order.get(oid) == "rate_or_timing_or_refund_deviation", oid


# ---------------------------------------------------------------------------
# Test 8 -- orphan and short_settlement: zero candidates found, correctly so
# ---------------------------------------------------------------------------

def test_orphan_and_short_settlement_excluded(fast_path_result, frozen_ground_truth):
    reason_by_order = _reason_by_order_id(fast_path_result)
    short_ids = _order_ids_for_category(frozen_ground_truth, "short_settlement")
    assert len(short_ids) >= 1
    for oid in short_ids:
        assert oid not in fast_path_result.resolved_order_ids, oid
        assert reason_by_order.get(oid) == "no_bank_candidate", oid

    orphan_entries = [e for e in frozen_ground_truth if e.category == "orphan"]
    assert len(orphan_entries) >= 1
    resolved_utrs = {g.utr for g in fast_path_result.resolved}
    for entry in orphan_entries:
        utr = entry.order_id[len("UNMATCHED_BANK_"):]
        assert utr not in resolved_utrs, entry.order_id


# ---------------------------------------------------------------------------
# Test 9 -- adversarial traps never auto-matched
# ---------------------------------------------------------------------------

def test_adversarial_traps_not_resolved(fast_path_result, frozen_ground_truth):
    trap_entries = [e for e in frozen_ground_truth if e.category == "adversarial_trap"]
    assert len(trap_entries) >= 5
    resolved_utrs = {g.utr for g in fast_path_result.resolved}
    for entry in trap_entries:
        utr = entry.order_id[len("UNMATCHED_BANK_"):]
        assert utr not in resolved_utrs, entry.order_id


# ---------------------------------------------------------------------------
# Test 11 -- duplicate_credit caught specifically by the cardinality guardrail
# ---------------------------------------------------------------------------

def test_duplicate_credit_caught_by_cardinality_guardrail(fast_path_result, frozen_ground_truth):
    dup_ids = _order_ids_for_category(frozen_ground_truth, "duplicate_credit")
    assert len(dup_ids) >= 1
    for oid in dup_ids:
        assert oid not in fast_path_result.resolved_order_ids, oid
        discrepancy = next(d for d in fast_path_result.discrepancies if oid in d.order_ids)
        assert discrepancy.reason == "ambiguous_multiple_candidates", oid
        assert discrepancy.candidate_count == 2, oid


# ---------------------------------------------------------------------------
# Test 13 -- a full utr_batch group resolves together, as a whole
# ---------------------------------------------------------------------------

def test_utr_batch_group_resolves_as_whole(fast_path_result, frozen_ground_truth):
    batch_ids = set(_order_ids_for_category(frozen_ground_truth, "utr_batch"))
    assert len(batch_ids) >= 2
    for oid in batch_ids:
        assert oid in fast_path_result.resolved_order_ids, oid
    multi_member_groups = [g for g in fast_path_result.resolved if len(g.order_ids) > 1]
    assert any(set(g.order_ids) & batch_ids for g in multi_member_groups)


# ---------------------------------------------------------------------------
# Synthetic fixture builder for the targeted unit tests below
# ---------------------------------------------------------------------------

def _build_upi_group(
    order_id: str,
    utr: str,
    narration: str,
    order_date: date,
    settlement_date: date,
    bank_value_date: date | None = None,
    gross: Decimal = Decimal("1000.00"),
    bank_utr: str | None = None,
) -> tuple[InternalOrder, GatewaySettlement, BankStatementLine]:
    mdr, gst, tds, net = _compute_settlement_fields(gross, "upi")
    order = InternalOrder(
        order_id=order_id,
        gross_amount=gross,
        customer_id=f"CUST_{order_id}",
        payment_method="upi",
        timestamp=datetime(order_date.year, order_date.month, order_date.day, 10, 0, 0),
    )
    settlement = GatewaySettlement(
        payment_id=f"PAY_{order_id}",
        order_id=order_id,
        gross_amount=gross,
        payment_method="upi",
        mdr=mdr,
        gst_on_mdr=gst,
        tds_194o=tds,
        net_amount=net,
        utr=utr,
        settlement_date=settlement_date,
        is_international=False,
    )
    bank_line = BankStatementLine(
        utr=bank_utr if bank_utr is not None else utr,
        credited_amount=net,
        value_date=bank_value_date if bank_value_date is not None else settlement_date,
        narration=narration,
    )
    return order, settlement, bank_line


# ---------------------------------------------------------------------------
# Test 10 -- near-identical narration ambiguity routes to the queue
# ---------------------------------------------------------------------------

def test_near_identical_narration_ambiguity_routes_to_queue():
    order, settlement, bank_line_1 = _build_upi_group(
        "ORD_AMBIG1",
        "UTR9988776655",
        "TXN/RZPY/88776655/CONF",
        order_date=date(2025, 1, 6),
        settlement_date=date(2025, 1, 7),
    )
    # Same amount, same date, narration differs by a single digit.
    bank_line_2 = BankStatementLine(
        utr="UTR0000000001",
        credited_amount=bank_line_1.credited_amount,
        value_date=bank_line_1.value_date,
        narration="TXN/RZPY/88776654/CONF",
    )

    orders_df = orders_to_frame([order])
    settlements_df = settlements_to_frame([settlement])
    bank_df = bank_lines_to_frame([bank_line_1, bank_line_2])

    result = run_fast_path(orders_df, settlements_df, bank_df)
    assert "ORD_AMBIG1" not in result.resolved_order_ids
    discrepancy = next(d for d in result.discrepancies if "ORD_AMBIG1" in d.order_ids)
    assert discrepancy.reason == "ambiguous_multiple_candidates"
    assert discrepancy.candidate_count == 2


# ---------------------------------------------------------------------------
# Test 12 -- weekend-crossing window: business-day calendar required
# ---------------------------------------------------------------------------

def test_weekend_crossing_window_regression():
    settlement_date = date(2025, 1, 9)  # Thursday
    bank_value_date = date(2025, 1, 13)  # Monday -- crosses the weekend
    raw_calendar_days = abs((bank_value_date - settlement_date).days)
    assert raw_calendar_days == 4, "test setup assumption broke"
    assert business_days_between(settlement_date, bank_value_date) == 2

    order, settlement, bank_line = _build_upi_group(
        "ORD_WKND1",
        "UTR1231231237",
        "TXN/RZPY/31231237/CONF",
        order_date=date(2025, 1, 8),  # Wednesday
        settlement_date=settlement_date,
        bank_value_date=bank_value_date,
    )
    orders_df = orders_to_frame([order])
    settlements_df = settlements_to_frame([settlement])
    bank_df = bank_lines_to_frame([bank_line])

    result = run_fast_path(orders_df, settlements_df, bank_df)

    # A naive +/-2 CALENDAR-day window would have rejected this (4 > 2).
    assert raw_calendar_days > 2
    # Our +/-2 BUSINESS-day window correctly accepts it.
    assert "ORD_WKND1" in result.resolved_order_ids
    resolved = next(g for g in result.resolved if "ORD_WKND1" in g.order_ids)
    assert resolved.match_method == "fuzzy_fallback"


# ---------------------------------------------------------------------------
# Test 14 -- phase 1 (exact token) and phase 2 (fuzzy fallback) both actually
# execute, not just phase 1 happening to cover everything
# ---------------------------------------------------------------------------

def test_narration_phase1_and_phase2_both_exercised():
    order_date = date(2025, 1, 6)
    settlement_date = date(2025, 1, 7)

    order_a, settlement_a, bank_a = _build_upi_group(
        "ORD_PHASE_A",
        "UTR1112223334",
        "NEFT-UTR1112223334-SETTLE",  # full token -- phase 1
        order_date=order_date,
        settlement_date=settlement_date,
    )
    order_b, settlement_b, bank_b = _build_upi_group(
        "ORD_PHASE_B",
        "UTR5556667778",
        "TXN/RZPY/56667778/CONF",  # truncated, no UTR prefix -- forces phase 2
        order_date=order_date,
        settlement_date=settlement_date,
    )

    orders_df = orders_to_frame([order_a, order_b])
    settlements_df = settlements_to_frame([settlement_a, settlement_b])
    bank_df = bank_lines_to_frame([bank_a, bank_b])

    result = run_fast_path(orders_df, settlements_df, bank_df)
    method_by_order = {oid: g.match_method for g in result.resolved for oid in g.order_ids}
    assert method_by_order.get("ORD_PHASE_A") == "exact_token"
    assert method_by_order.get("ORD_PHASE_B") == "fuzzy_fallback"


# ---------------------------------------------------------------------------
# Test 15 -- bank_line.utr column is never consulted as a matching key
# ---------------------------------------------------------------------------

def test_bank_line_utr_column_not_consulted():
    order, settlement, bank_line = _build_upi_group(
        "ORD_UTRCOL1",
        "UTR7777777777",
        "NEFT-UTR7777777777-SETTLE",
        order_date=date(2025, 1, 6),
        settlement_date=date(2025, 1, 7),
        bank_utr="UTR0000000000",  # deliberately wrong; narration is correct
    )
    orders_df = orders_to_frame([order])
    settlements_df = settlements_to_frame([settlement])
    bank_df = bank_lines_to_frame([bank_line])

    result = run_fast_path(orders_df, settlements_df, bank_df)
    assert "ORD_UTRCOL1" in result.resolved_order_ids
    resolved = next(g for g in result.resolved if "ORD_UTRCOL1" in g.order_ids)
    assert resolved.match_method == "exact_token"


# ---------------------------------------------------------------------------
# Test 16 -- rate revalidation uses exact integer-paise arithmetic
# ---------------------------------------------------------------------------

def test_rate_revalidation_uses_integer_paise_exactly():
    gross_paise = 1500  # Rs 15.00 -- TDS 0.1% lands exactly on a half-cent boundary
    naive_float_tds_rupees = round(float(from_paise(gross_paise)) * float(TDS_RATE), 2)
    assert naive_float_tds_rupees == 0.01, "trap assumption broke: float no longer mis-rounds this value"

    _, _, tds_paise = expected_mdr_gst_tds_paise(gross_paise, "upi")
    assert tds_paise == 2  # Decimal ROUND_HALF_UP: 15.00 * 0.001 = 0.015 -> rounds to 0.02
    assert tds_paise != to_paise(Decimal(str(naive_float_tds_rupees)))

    mdr, gst, tds, net = _compute_settlement_fields(Decimal("15.00"), "upi")
    settlement = GatewaySettlement(
        payment_id="PAY_RATECHK",
        order_id="ORD_RATECHK",
        gross_amount=Decimal("15.00"),
        payment_method="upi",
        mdr=mdr,
        gst_on_mdr=gst,
        tds_194o=tds,
        net_amount=net,
        utr="UTR0000000099",
        settlement_date=date(2025, 1, 7),
        is_international=False,
    )
    settlements_df = settlements_to_frame([settlement])
    frame = compute_expected_rate_frame(settlements_df)
    assert frame.schema["expected_mdr_paise"] == pl.Int64
    assert frame.schema["expected_gst_paise"] == pl.Int64
    assert frame.schema["expected_tds_paise"] == pl.Int64
    row = frame.filter(pl.col("order_id") == "ORD_RATECHK")
    assert row["expected_tds_paise"][0] == 2
