"""Layer 1 tests for src/data/generator.py -- the synthetic dataset generator.
Written before implementation per CLAUDE.md's build protocol.
"""
import json
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pytest

from src.common.money import to_paise
from src.data.generator import (
    CATEGORY_COUNTS,
    GST_RATE,
    TDS_RATE,
    _compute_settlement_fields,
    generate_batch,
    write_batch,
)

SEED = 42
NUM_RECORDS = 100


@pytest.fixture(scope="module")
def batch():
    return generate_batch(num_records=NUM_RECORDS, seed=SEED)


@pytest.fixture(scope="module")
def raw_json_text(batch, tmp_path_factory):
    out_dir = tmp_path_factory.mktemp("batch_for_raw_text")
    write_batch(batch, out_dir)
    texts = {}
    for name in (
        "internal_orders.json",
        "gateway_settlement.json",
        "bank_statement.json",
        "ground_truth.json",
    ):
        texts[name] = (out_dir / name).read_text(encoding="utf-8")
    return texts


# ---------------------------------------------------------------------------
# Criterion 1 -- generator runs, produces all 4 files
# ---------------------------------------------------------------------------

def test_generator_produces_all_four_files(batch, tmp_path):
    write_batch(batch, tmp_path)
    for name in (
        "internal_orders.json",
        "gateway_settlement.json",
        "bank_statement.json",
        "ground_truth.json",
    ):
        f = tmp_path / name
        assert f.exists(), f"missing {name}"
        content = json.loads(f.read_text(encoding="utf-8"))
        assert len(content) > 0


# ---------------------------------------------------------------------------
# Criterion 2 -- same seed produces byte-identical output
# ---------------------------------------------------------------------------

def test_generator_deterministic_same_seed(tmp_path):
    batch_a = generate_batch(num_records=NUM_RECORDS, seed=SEED)
    batch_b = generate_batch(num_records=NUM_RECORDS, seed=SEED)

    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"
    write_batch(batch_a, dir_a)
    write_batch(batch_b, dir_b)

    for name in (
        "internal_orders.json",
        "gateway_settlement.json",
        "bank_statement.json",
        "ground_truth.json",
    ):
        assert (dir_a / name).read_bytes() == (dir_b / name).read_bytes(), (
            f"{name} differs between two runs of the same seed"
        )


def test_generator_different_seeds_produce_different_output(tmp_path):
    batch_a = generate_batch(num_records=NUM_RECORDS, seed=1)
    batch_b = generate_batch(num_records=NUM_RECORDS, seed=2)
    dir_a = tmp_path / "seed1"
    dir_b = tmp_path / "seed2"
    write_batch(batch_a, dir_a)
    write_batch(batch_b, dir_b)
    assert (dir_a / "internal_orders.json").read_bytes() != (
        dir_b / "internal_orders.json"
    ).read_bytes()


# ---------------------------------------------------------------------------
# Criterion 3 -- ground_truth.json has 100 entries, category counts match spec
# ---------------------------------------------------------------------------

def test_ground_truth_has_100_entries(batch):
    assert len(batch.ground_truth) == 100


def test_ground_truth_category_counts_match_spec(batch):
    counts = Counter(entry.category for entry in batch.ground_truth)
    for category, expected_count in CATEGORY_COUNTS.items():
        assert abs(counts[category] - expected_count) <= 1, (
            f"{category}: expected {expected_count} (+/-1), got {counts[category]}"
        )
    assert sum(CATEGORY_COUNTS.values()) == 100


# ---------------------------------------------------------------------------
# Criterion 4 -- agent_resolved entries have non-null fields; others null
# ---------------------------------------------------------------------------

def test_agent_resolved_entries_have_populated_root_cause_and_delta(batch):
    for entry in batch.ground_truth:
        if entry.expected_resolution == "agent_resolved":
            assert entry.expected_root_cause_code is not None, entry.order_id
            assert entry.expected_delta_paise is not None, entry.order_id


def test_fast_path_and_honest_exception_entries_have_null_fields(batch):
    for entry in batch.ground_truth:
        if entry.expected_resolution in ("fast_path", "honest_exception"):
            assert entry.expected_root_cause_code is None, entry.order_id
            assert entry.expected_delta_paise is None, entry.order_id


# ---------------------------------------------------------------------------
# Criterion 5 -- no UPI settlement has nonzero mdr or gst_on_mdr
# ---------------------------------------------------------------------------

def test_no_upi_settlement_has_nonzero_mdr_or_gst(batch):
    upi_settlements = [s for s in batch.settlements if s.payment_method == "upi"]
    assert len(upi_settlements) > 0, "test is vacuous with no UPI settlements generated"
    for s in upi_settlements:
        assert s.mdr == Decimal("0.00")
        assert s.gst_on_mdr == Decimal("0.00")


# ---------------------------------------------------------------------------
# Criterion 6 -- TDS is 0.1%, never 1%
# ---------------------------------------------------------------------------

def test_tds_rate_constant_is_point_one_percent():
    assert TDS_RATE == Decimal("0.001")


def test_every_settlement_tds_computed_at_point_one_percent(batch):
    # missing_tax_line/MISSING_TDS records deliberately zero this line as the
    # anomaly under test (see docs/plan.md Layer 1 Addendum A1) -- excluded
    # here since their zeroing is intentional, not a rate-computation bug.
    tds_omitted_order_ids = {
        e.order_id
        for e in batch.ground_truth
        if e.category == "missing_tax_line" and e.expected_root_cause_code == "MISSING_TDS"
    }
    checked_any = False
    for s in batch.settlements:
        if s.order_id in tds_omitted_order_ids:
            assert s.tds_194o == Decimal("0.00"), s.payment_id
            continue
        expected_tds = (s.gross_amount * TDS_RATE).quantize(Decimal("0.01"))
        assert s.tds_194o == expected_tds, s.payment_id
        # Explicitly rule out the old 1% rate having been used instead.
        wrong_tds = (s.gross_amount * Decimal("0.01")).quantize(Decimal("0.01"))
        if expected_tds != wrong_tds:
            assert s.tds_194o != wrong_tds, s.payment_id
        checked_any = True
    assert checked_any, "test is vacuous with no non-omitted settlements to check"


def test_gst_rate_constant_is_eighteen_percent():
    assert GST_RATE == Decimal("0.18")


# ---------------------------------------------------------------------------
# Criterion 7 -- short_settlement and duplicate_credit present, distinguishable
# from orphan
# ---------------------------------------------------------------------------

def test_short_settlement_count(batch):
    counts = Counter(entry.category for entry in batch.ground_truth)
    assert counts["short_settlement"] == CATEGORY_COUNTS["short_settlement"]


def test_duplicate_credit_count(batch):
    counts = Counter(entry.category for entry in batch.ground_truth)
    assert counts["duplicate_credit"] == CATEGORY_COUNTS["duplicate_credit"]


def test_short_settlement_distinguishable_from_orphan(batch):
    # short_settlement: a real settlement exists for the UTR, but no bank
    # credit for it ever appears.
    short_settlement_entries = [
        e for e in batch.ground_truth if e.category == "short_settlement"
    ]
    settlement_utrs = {s.utr for s in batch.settlements}
    bank_utrs = {b.utr for b in batch.bank_lines}
    for entry in short_settlement_entries:
        matching_settlements = [s for s in batch.settlements if s.order_id == entry.order_id]
        assert len(matching_settlements) == 1
        utr = matching_settlements[0].utr
        assert utr in settlement_utrs
        assert utr not in bank_utrs


def test_orphan_distinguishable_from_short_settlement(batch):
    # orphan: a bank credit exists whose UTR matches no settlement record at all.
    orphan_entries = [e for e in batch.ground_truth if e.category == "orphan"]
    settlement_utrs = {s.utr for s in batch.settlements}
    for entry in orphan_entries:
        assert entry.order_id.startswith("UNMATCHED_BANK_")
        utr = entry.order_id[len("UNMATCHED_BANK_"):]
        assert utr not in settlement_utrs
        assert any(b.utr == utr for b in batch.bank_lines)


def test_duplicate_credit_has_two_bank_lines_for_same_utr(batch):
    duplicate_entries = [e for e in batch.ground_truth if e.category == "duplicate_credit"]
    for entry in duplicate_entries:
        matching_settlements = [s for s in batch.settlements if s.order_id == entry.order_id]
        assert len(matching_settlements) == 1
        utr = matching_settlements[0].utr
        lines_for_utr = [b for b in batch.bank_lines if b.utr == utr]
        assert len(lines_for_utr) == 2, (
            f"duplicate_credit order {entry.order_id} should have exactly 2 bank "
            f"lines on UTR {utr}, found {len(lines_for_utr)}"
        )


# ---------------------------------------------------------------------------
# Criterion 8 -- at least 5 adversarial_trap records
# ---------------------------------------------------------------------------

def test_adversarial_trap_count_at_least_five(batch):
    counts = Counter(entry.category for entry in batch.ground_truth)
    assert counts["adversarial_trap"] >= 5


def test_adversarial_trap_entries_are_honest_exception(batch):
    trap_entries = [e for e in batch.ground_truth if e.category == "adversarial_trap"]
    assert len(trap_entries) > 0
    for entry in trap_entries:
        assert entry.expected_resolution == "honest_exception"
        assert entry.expected_root_cause_code is None
        assert entry.expected_delta_paise is None


# ---------------------------------------------------------------------------
# Criterion 9a -- no floats anywhere in generator output
# ---------------------------------------------------------------------------

def test_no_floats_in_raw_json_output(raw_json_text):
    for name, text in raw_json_text.items():
        parsed = json.loads(text)
        _assert_no_floats(parsed, path=name)


def _assert_no_floats(value, path):
    if isinstance(value, float):
        pytest.fail(f"float found in generator output at {path}: {value!r}")
    elif isinstance(value, dict):
        for k, v in value.items():
            _assert_no_floats(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            _assert_no_floats(v, f"{path}[{i}]")


# ---------------------------------------------------------------------------
# Criterion 9b -- net_amount invariant holds on every settlement record
# ---------------------------------------------------------------------------

def test_net_amount_invariant_holds_on_every_generated_settlement(batch):
    for s in batch.settlements:
        assert s.net_amount == s.gross_amount - s.mdr - s.gst_on_mdr - s.tds_194o, s.payment_id


# ---------------------------------------------------------------------------
# Criterion 9c -- orphans and short_settlements have zero valid match anywhere
# ---------------------------------------------------------------------------

def test_orphans_have_no_valid_match_anywhere_in_batch(batch):
    orphan_entries = [e for e in batch.ground_truth if e.category == "orphan"]
    order_ids = {o.order_id for o in batch.orders}
    for entry in orphan_entries:
        utr = entry.order_id[len("UNMATCHED_BANK_"):]
        assert not any(s.utr == utr for s in batch.settlements)


def test_short_settlements_have_no_valid_match_anywhere_in_batch(batch):
    short_entries = [e for e in batch.ground_truth if e.category == "short_settlement"]
    for entry in short_entries:
        settlement = next(s for s in batch.settlements if s.order_id == entry.order_id)
        assert not any(b.utr == settlement.utr for b in batch.bank_lines)


# ---------------------------------------------------------------------------
# Layer 1 Addendum A2 -- refund_clawback
# ---------------------------------------------------------------------------

def test_refund_clawback_count(batch):
    counts = Counter(entry.category for entry in batch.ground_truth)
    assert counts["refund_clawback"] == CATEGORY_COUNTS["refund_clawback"]


def test_refund_clawback_orders_have_refund_amount_set(batch):
    entries = [e for e in batch.ground_truth if e.category == "refund_clawback"]
    assert len(entries) > 0
    for entry in entries:
        order = next(o for o in batch.orders if o.order_id == entry.order_id)
        assert order.refund_amount is not None
        assert Decimal("0") < order.refund_amount < order.gross_amount
        assert entry.expected_root_cause_code == "REFUND_NO_MDR_REVERSAL"
        assert entry.expected_delta_paise == to_paise(order.refund_amount)


# ---------------------------------------------------------------------------
# Layer 1 Addendum A5 -- dates fixed in code, not wall-clock derived
# ---------------------------------------------------------------------------

def test_all_order_timestamps_within_fixed_batch_window(batch):
    from src.data.generator import BATCH_END, BATCH_START

    for order in batch.orders:
        assert BATCH_START <= order.timestamp.date() <= BATCH_END, order.order_id


# ---------------------------------------------------------------------------
# Gap fix 1 -- missing_tax_line: dedicated coverage of the Q1 resolution
# (docs/plan.md Layer 1 Addendum A1). A missing_tax_line record must (a) stay
# internally valid -- net_amount consistent with the real, uncorrected error
# -- while also (b) producing a genuine, catchable discrepancy when the
# zeroed line is recomputed against the actual GST/TDS rate constants.
# ---------------------------------------------------------------------------

def test_missing_tax_line_count(batch):
    counts = Counter(entry.category for entry in batch.ground_truth)
    assert counts["missing_tax_line"] == CATEGORY_COUNTS["missing_tax_line"]


def test_missing_tax_line_both_root_causes_reachable(batch):
    # Not just settable -- actually produced by the generator for this seed.
    root_causes = {
        e.expected_root_cause_code
        for e in batch.ground_truth
        if e.category == "missing_tax_line"
    }
    assert root_causes == {"MISSING_GST", "MISSING_TDS"}


def test_missing_tax_line_settlements_are_internally_consistent(batch):
    # (a) The record itself stays valid: net_amount reflects the real,
    # uncorrected shortfall, not a value inconsistent with the zeroed line.
    # GatewaySettlement's own model_validator already enforces this at
    # construction time; this test proves it holds for every missing_tax_line
    # record specifically, not just settlements in general.
    entries = [e for e in batch.ground_truth if e.category == "missing_tax_line"]
    assert len(entries) > 0
    for entry in entries:
        settlement = next(s for s in batch.settlements if s.order_id == entry.order_id)
        assert settlement.net_amount == (
            settlement.gross_amount - settlement.mdr - settlement.gst_on_mdr - settlement.tds_194o
        ), entry.order_id
        if entry.expected_root_cause_code == "MISSING_GST":
            assert settlement.gst_on_mdr == Decimal("0.00"), entry.order_id
            assert settlement.tds_194o != Decimal("0.00"), entry.order_id
        else:
            assert settlement.tds_194o == Decimal("0.00"), entry.order_id
            assert settlement.gst_on_mdr != Decimal("0.00"), entry.order_id


def test_missing_tax_line_produces_real_discrepancy_against_recomputed_tax_rules(batch):
    # (b) Recomputing the omitted line from the actual GST/TDS constants
    # against this settlement's own gross/mdr must diverge from the recorded
    # (zeroed) value by exactly expected_delta_paise -- proving this is a
    # genuine, quantifiable discrepancy, not a cosmetically-valid record with
    # nothing for a diagnostic tool to actually catch.
    entries = [e for e in batch.ground_truth if e.category == "missing_tax_line"]
    assert len(entries) > 0
    for entry in entries:
        settlement = next(s for s in batch.settlements if s.order_id == entry.order_id)
        if entry.expected_root_cause_code == "MISSING_GST":
            recomputed = (settlement.mdr * GST_RATE).quantize(Decimal("0.01"))
            recorded = settlement.gst_on_mdr
        else:
            recomputed = (settlement.gross_amount * TDS_RATE).quantize(Decimal("0.01"))
            recorded = settlement.tds_194o
        assert recomputed > 0, entry.order_id  # there must be something real to catch
        assert recorded != recomputed, entry.order_id
        assert to_paise(recomputed - recorded) == entry.expected_delta_paise, entry.order_id


# ---------------------------------------------------------------------------
# Gap fix 2 -- refund_clawback: MDR is structurally never reversed by a
# refund (Q2, docs/plan.md Layer 1 Addendum A2). There is no separate
# "mdr_reversed" field or refund_events.json in this implementation -- A2
# modeled the rule structurally, via GatewaySettlement being generated from
# gross_amount alone, unaffected by InternalOrder.refund_amount. This test
# proves that structural invariant directly: a refunded order's settlement
# carries exactly the MDR/GST it would have carried had no refund occurred.
# ---------------------------------------------------------------------------

def test_refund_clawback_settlement_mdr_never_reduced_by_refund(batch):
    entries = [e for e in batch.ground_truth if e.category == "refund_clawback"]
    assert len(entries) > 0
    for entry in entries:
        order = next(o for o in batch.orders if o.order_id == entry.order_id)
        settlement = next(s for s in batch.settlements if s.order_id == entry.order_id)
        assert order.refund_amount is not None, entry.order_id

        unaffected_mdr, unaffected_gst, unaffected_tds, unaffected_net = _compute_settlement_fields(
            order.gross_amount, settlement.payment_method
        )
        assert settlement.mdr == unaffected_mdr, entry.order_id
        assert settlement.gst_on_mdr == unaffected_gst, entry.order_id
        assert settlement.gross_amount == order.gross_amount, entry.order_id
        # The refund reduces nothing on the gateway side -- it is purely an
        # InternalOrder-level fact the ledger must reconcile against later.
        assert settlement.net_amount == unaffected_net, entry.order_id


# ---------------------------------------------------------------------------
# Gap fix 3 -- ground_truth.json keying scheme (Q4, docs/plan.md Layer 1
# Addendum A4). GroundTruthEntry has no separate utr field -- order-less
# anomalies are keyed entirely through the UNMATCHED_BANK_<utr> order_id
# scheme. This proves that scheme is actually applied correctly: synthetic
# keys never collide with a real order_id, and categories anchored to a real
# order actually reference one that exists.
# ---------------------------------------------------------------------------

def test_order_less_categories_use_synthetic_keys_never_colliding_with_real_orders(batch):
    real_order_ids = {o.order_id for o in batch.orders}
    order_less_entries = [
        e for e in batch.ground_truth if e.category in ("orphan", "adversarial_trap")
    ]
    assert len(order_less_entries) > 0
    for entry in order_less_entries:
        assert entry.order_id.startswith("UNMATCHED_BANK_"), entry.order_id
        assert entry.order_id not in real_order_ids, entry.order_id
        utr = entry.order_id[len("UNMATCHED_BANK_"):]
        assert any(b.utr == utr for b in batch.bank_lines), entry.order_id


def test_order_anchored_categories_reference_a_real_order(batch):
    real_order_ids = {o.order_id for o in batch.orders}
    order_anchored_entries = [
        e
        for e in batch.ground_truth
        if e.category
        in ("clean_match", "utr_batch", "cutoff_drift", "fee_drift", "missing_tax_line",
            "short_settlement", "duplicate_credit", "refund_clawback")
    ]
    assert len(order_anchored_entries) > 0
    for entry in order_anchored_entries:
        assert entry.order_id in real_order_ids, entry.order_id
        assert not entry.order_id.startswith("UNMATCHED_BANK_"), entry.order_id


# ---------------------------------------------------------------------------
# Gap fix 4 -- fee_drift's is_international flag must actually be set by the
# generator, not merely accepted by the model (Q3, docs/plan.md Layer 1
# Addendum A3). Without this, INTL_MARKUP is a reachable RootCauseCode value
# in principle but dead in practice.
# ---------------------------------------------------------------------------

def test_fee_drift_is_international_flag_actually_used_by_generator(batch):
    fee_drift_entries = [e for e in batch.ground_truth if e.category == "fee_drift"]
    assert len(fee_drift_entries) > 0

    international_root_causes = set()
    domestic_root_causes = set()
    for entry in fee_drift_entries:
        settlement = next(s for s in batch.settlements if s.order_id == entry.order_id)
        if settlement.is_international:
            international_root_causes.add(entry.expected_root_cause_code)
        else:
            domestic_root_causes.add(entry.expected_root_cause_code)

    assert "INTL_MARKUP" in international_root_causes, (
        "no fee_drift settlement had is_international=True -- INTL_MARKUP is unreachable"
    )
    assert "AMEX_SURCHARGE" in domestic_root_causes, (
        "no fee_drift settlement had is_international=False -- AMEX_SURCHARGE is unreachable"
    )
