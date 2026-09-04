"""Layer 8 tests for evaluate.py (the evaluation harness).

Deterministic-block tests (1-9) run the real pipeline against the frozen
challenge_batch_100 dataset over a real Postgres test database (see
tests/conftest.py) -- CLAUDE.md forbids mocking the ledger layer. They make
zero live model calls: every discrepancy-queue record in the frozen dataset
is already covered by data/agent_runs/layer4_test_cache.jsonl, the same
guarantee tests/test_api.py's test_trigger_batch_run_frozen_makes_zero_live_agent_calls
relies on.

Agent-block tests (10-16) test the scoring/variance/logging/replay
machinery only, via a small hand-built fixture and an injectable stub
diagnose_fn -- the same test-isolation seam already established by
tests/test_api.py's stub for the "seed" source path. They never touch the
network, and Layer 8 deliberately does not spend live model budget running
the real 3x sweep in this suite (see evaluate.py's module docstring); that
happens separately, when a human runs `python evaluate.py` for real.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

from evaluate import (
    AgentBlockReport,
    DeterministicReport,
    delta_within_tolerance,
    format_agent_block_report,
    format_deterministic_report,
    gt_key_for_record,
    resolutions_from_log,
    run_agent_block_once,
    run_agent_block_resuming,
    run_deterministic_block,
    score_agent_runs,
)
from src.agent.discrepancy import BankCredit, DiscrepancyRecord, OrderContext
from src.agent.resolution import AgentResolution
from src.data.models import BankStatementLine, GatewaySettlement, GroundTruthEntry, InternalOrder

FROZEN_DIR = Path(__file__).resolve().parents[1] / "data" / "challenge_batch_100"


# ---------------------------------------------------------------------------
# Fixtures: the frozen dataset, parsed once per module.
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


# ---------------------------------------------------------------------------
# Tests 1-9: deterministic block, real pipeline, real Postgres, zero live calls.
# ---------------------------------------------------------------------------

def test_deterministic_block_counts_match_ground_truth(
    db_session, batch_run_id, frozen_orders, frozen_settlements, frozen_bank_lines, frozen_ground_truth
):
    report = run_deterministic_block(
        db_session, batch_run_id, frozen_orders, frozen_settlements, frozen_bank_lines, frozen_ground_truth
    )
    assert report.fast_path_count == 63
    assert report.agent_resolved_count == 20
    assert report.honest_exception_count == 17


def test_deterministic_block_zero_false_auto_resolutions(
    db_session, batch_run_id, frozen_orders, frozen_settlements, frozen_bank_lines, frozen_ground_truth
):
    report = run_deterministic_block(
        db_session, batch_run_id, frozen_orders, frozen_settlements, frozen_bank_lines, frozen_ground_truth
    )
    assert report.false_auto_resolutions == 0
    assert report.false_auto_resolution_ids == []


def test_deterministic_block_adversarial_traps_5_of_5(
    db_session, batch_run_id, frozen_orders, frozen_settlements, frozen_bank_lines, frozen_ground_truth
):
    report = run_deterministic_block(
        db_session, batch_run_id, frozen_orders, frozen_settlements, frozen_bank_lines, frozen_ground_truth
    )
    assert report.adversarial_traps_total == 5
    assert report.adversarial_traps_caught == 5


def test_deterministic_block_duplicate_credits_2_of_2(
    db_session, batch_run_id, frozen_orders, frozen_settlements, frozen_bank_lines, frozen_ground_truth
):
    report = run_deterministic_block(
        db_session, batch_run_id, frozen_orders, frozen_settlements, frozen_bank_lines, frozen_ground_truth
    )
    assert report.duplicate_credits_total == 2
    assert report.duplicate_credits_caught == 2


def test_deterministic_block_ledger_balance_pass(
    db_session, batch_run_id, frozen_orders, frozen_settlements, frozen_bank_lines, frozen_ground_truth
):
    report = run_deterministic_block(
        db_session, batch_run_id, frozen_orders, frozen_settlements, frozen_bank_lines, frozen_ground_truth
    )
    assert report.ledger_balance_pass is True
    total_row = next(r for r in report.trial_balance_rows if r["account_code"] == "TOTAL")
    assert total_row["net_balance_paise"] == 0


def test_deterministic_block_trial_balance_full_table_present(
    db_session, batch_run_id, frozen_orders, frozen_settlements, frozen_bank_lines, frozen_ground_truth
):
    report = run_deterministic_block(
        db_session, batch_run_id, frozen_orders, frozen_settlements, frozen_bank_lines, frozen_ground_truth
    )
    account_codes = {row["account_code"] for row in report.trial_balance_rows}
    expected_accounts = {
        "CASH",
        "CASH_IN_TRANSIT_UTR",
        "AR_GATEWAY_CLEARING",
        "REVENUE_GROSS",
        "MDR_EXPENSE",
        "GST_ITC_RECEIVABLE",
        "TDS_194O_CREDIT",
        "SUSPENSE_UNRESOLVED",
        "TOTAL",
    }
    assert expected_accounts.issubset(account_codes)


def test_deterministic_block_paise_round_trip_zero_drift(
    db_session, batch_run_id, frozen_orders, frozen_settlements, frozen_bank_lines, frozen_ground_truth
):
    report = run_deterministic_block(
        db_session, batch_run_id, frozen_orders, frozen_settlements, frozen_bank_lines, frozen_ground_truth
    )
    assert report.paise_round_trip_drift == 0


def test_deterministic_block_reproducible_across_two_runs(
    db_session, frozen_orders, frozen_settlements, frozen_bank_lines, frozen_ground_truth
):
    run_id_a = uuid.uuid4()
    run_id_b = uuid.uuid4()
    report_a = run_deterministic_block(
        db_session, run_id_a, frozen_orders, frozen_settlements, frozen_bank_lines, frozen_ground_truth
    )
    report_b = run_deterministic_block(
        db_session, run_id_b, frozen_orders, frozen_settlements, frozen_bank_lines, frozen_ground_truth
    )
    assert report_a.fast_path_count == report_b.fast_path_count
    assert report_a.agent_resolved_count == report_b.agent_resolved_count
    assert report_a.honest_exception_count == report_b.honest_exception_count
    assert report_a.false_auto_resolutions == report_b.false_auto_resolutions
    assert report_a.adversarial_traps_caught == report_b.adversarial_traps_caught
    assert report_a.duplicate_credits_caught == report_b.duplicate_credits_caught
    assert report_a.ledger_balance_pass == report_b.ledger_balance_pass
    assert report_a.paise_round_trip_drift == report_b.paise_round_trip_drift

    def _rows_by_account(report):
        return {row["account_code"]: (row["debit_total_paise"], row["credit_total_paise"]) for row in report.trial_balance_rows}

    assert _rows_by_account(report_a) == _rows_by_account(report_b)


def test_deterministic_block_never_calls_live_model(
    db_session, batch_run_id, frozen_orders, frozen_settlements, frozen_bank_lines, frozen_ground_truth, monkeypatch
):
    import src.agent.graph as graph_module

    def _fail_if_called(record):
        raise AssertionError("live diagnose_discrepancy was called for the frozen (fully-cached) batch")

    monkeypatch.setattr(graph_module, "diagnose_discrepancy", _fail_if_called)

    report = run_deterministic_block(
        db_session, batch_run_id, frozen_orders, frozen_settlements, frozen_bank_lines, frozen_ground_truth
    )
    assert report.fast_path_count == 63


# ---------------------------------------------------------------------------
# Small hand-built fixture for the agent-block tests: 2 agent_resolved
# records, 1 honest_exception record. Independent of the frozen dataset --
# these tests are about the scoring/variance machinery, not the real batch.
# ---------------------------------------------------------------------------

def _tiny_records() -> list[DiscrepancyRecord]:
    return [
        DiscrepancyRecord(
            discrepancy_reason="fee_drift",
            order_context=OrderContext(
                order_id="ORD_A",
                gross_amount_paise=100_000,
                payment_method="amex",
                timestamp="2025-01-06T10:00:00+05:30",
                refund_amount_paise=None,
            ),
        ),
        DiscrepancyRecord(
            discrepancy_reason="missing_tax_line",
            order_context=OrderContext(
                order_id="ORD_B",
                gross_amount_paise=50_000,
                payment_method="credit_card",
                timestamp="2025-01-07T11:00:00+05:30",
                refund_amount_paise=None,
            ),
        ),
        DiscrepancyRecord(
            discrepancy_reason="unmatched_bank_line",
            bank_credits=[
                BankCredit(
                    utr="UTR_C", credited_amount_paise=20_000, value_date="2025-01-10", narration="TEST-CREDIT"
                )
            ],
        ),
    ]


def _tiny_ground_truth() -> list[GroundTruthEntry]:
    return [
        GroundTruthEntry(
            order_id="ORD_A",
            category="fee_drift",
            expected_resolution="agent_resolved",
            expected_root_cause_code="AMEX_SURCHARGE",
            expected_delta_paise=1000,
            notes="",
        ),
        GroundTruthEntry(
            order_id="ORD_B",
            category="missing_tax_line",
            expected_resolution="agent_resolved",
            expected_root_cause_code="MISSING_GST",
            expected_delta_paise=200,
            notes="",
        ),
        GroundTruthEntry(
            order_id="UNMATCHED_BANK_UTR_C",
            category="orphan",
            expected_resolution="honest_exception",
            expected_root_cause_code=None,
            expected_delta_paise=None,
            notes="",
        ),
    ]


def _resolution(root_cause_code: str, delta_paise: int = 0) -> AgentResolution:
    return AgentResolution(
        root_cause_code=root_cause_code,
        quantified_delta_paise=delta_paise,
        evidence_tool_calls=[],
        confidence_note="stub",
    )


def _make_stub(answers: dict[str, AgentResolution]):
    def _fn(record: DiscrepancyRecord):
        return answers[gt_key_for_record(record)], {"hop_count": 1, "tokens_used": 10}

    return _fn


# ---------------------------------------------------------------------------
# Test 10: the agent block reports a range, never collapses to one number
# ---------------------------------------------------------------------------

def test_agent_block_reports_range_not_single_number(tmp_path):
    records = _tiny_records()
    ground_truth = _tiny_ground_truth()

    run1 = run_agent_block_once(
        records,
        _make_stub(
            {
                "ORD_A": _resolution("AMEX_SURCHARGE", 1000),
                "ORD_B": _resolution("UNRESOLVED", 0),
                "UNMATCHED_BANK_UTR_C": _resolution("UNRESOLVED", 0),
            }
        ),
        tmp_path / "run1.jsonl",
        logic_version=1,
    )
    run2 = run_agent_block_once(
        records,
        _make_stub(
            {
                "ORD_A": _resolution("AMEX_SURCHARGE", 1000),
                "ORD_B": _resolution("MISSING_GST", 200),
                "UNMATCHED_BANK_UTR_C": _resolution("UNRESOLVED", 0),
            }
        ),
        tmp_path / "run2.jsonl",
        logic_version=1,
    )
    run3 = run_agent_block_once(
        records,
        _make_stub(
            {
                "ORD_A": _resolution("AMEX_SURCHARGE", 1000),
                "ORD_B": _resolution("MISSING_GST", 200),
                "UNMATCHED_BANK_UTR_C": _resolution("UNRESOLVED", 0),
            }
        ),
        tmp_path / "run3.jsonl",
        logic_version=1,
    )

    report = score_agent_runs([run1, run2, run3], ground_truth)
    assert report.resolved_per_run == [1, 2, 2]
    assert len(set(report.resolved_per_run)) > 1, "real variance must survive into the report, not average away"

    text = format_agent_block_report(report)
    assert "[1, 2, 2]" in text
    assert "Agent block" in text


# ---------------------------------------------------------------------------
# Test 11: root_cause_code correctness and delta correctness are scored
# independently against ground truth
# ---------------------------------------------------------------------------

def test_agent_block_correctness_scored_independently(tmp_path):
    records = _tiny_records()
    ground_truth = _tiny_ground_truth()

    # run3: ORD_B gets the WRONG root cause but happens to report the RIGHT
    # delta -- proves root-cause correctness and delta correctness are
    # scored as separate metrics, not coupled.
    run1 = run_agent_block_once(
        records,
        _make_stub(
            {
                "ORD_A": _resolution("AMEX_SURCHARGE", 1000),
                "ORD_B": _resolution("UNRESOLVED", 0),
                "UNMATCHED_BANK_UTR_C": _resolution("UNRESOLVED", 0),
            }
        ),
        tmp_path / "run1.jsonl",
        logic_version=1,
    )
    run2 = run_agent_block_once(
        records,
        _make_stub(
            {
                "ORD_A": _resolution("AMEX_SURCHARGE", 1000),
                "ORD_B": _resolution("MISSING_GST", 200),
                "UNMATCHED_BANK_UTR_C": _resolution("UNRESOLVED", 0),
            }
        ),
        tmp_path / "run2.jsonl",
        logic_version=1,
    )
    run3 = run_agent_block_once(
        records,
        _make_stub(
            {
                "ORD_A": _resolution("AMEX_SURCHARGE", 1000),
                "ORD_B": _resolution("AMEX_SURCHARGE", 200),  # wrong code, correct delta
                "UNMATCHED_BANK_UTR_C": _resolution("UNRESOLVED", 0),
            }
        ),
        tmp_path / "run3.jsonl",
        logic_version=1,
    )

    report = score_agent_runs([run1, run2, run3], ground_truth)
    assert report.correct_root_cause_per_run == [1, 2, 1]
    assert report.correct_delta_per_run == [1, 2, 2]


# ---------------------------------------------------------------------------
# Test 12: honest-exception consistency across runs, both directions
# ---------------------------------------------------------------------------

def test_agent_block_honest_exception_flagged_consistent(tmp_path):
    records = _tiny_records()
    ground_truth = _tiny_ground_truth()
    stub_answers = {
        "ORD_A": _resolution("AMEX_SURCHARGE", 1000),
        "ORD_B": _resolution("MISSING_GST", 200),
        "UNMATCHED_BANK_UTR_C": _resolution("UNRESOLVED", 0),
    }
    runs = [
        run_agent_block_once(records, _make_stub(stub_answers), tmp_path / f"run{i}.jsonl", logic_version=1)
        for i in range(3)
    ]
    report = score_agent_runs(runs, ground_truth)
    assert report.honest_exception_consistent is True
    assert report.honest_exception_inconsistent_ids == []


def test_agent_block_honest_exception_flagged_inconsistent(tmp_path):
    records = _tiny_records()
    ground_truth = _tiny_ground_truth()
    base_answers = {
        "ORD_A": _resolution("AMEX_SURCHARGE", 1000),
        "ORD_B": _resolution("MISSING_GST", 200),
    }
    run1 = run_agent_block_once(
        records,
        _make_stub({**base_answers, "UNMATCHED_BANK_UTR_C": _resolution("UNRESOLVED", 0)}),
        tmp_path / "run1.jsonl",
        logic_version=1,
    )
    run2 = run_agent_block_once(
        records,
        _make_stub({**base_answers, "UNMATCHED_BANK_UTR_C": _resolution("UNRESOLVED", 0)}),
        tmp_path / "run2.jsonl",
        logic_version=1,
    )
    # run3 force-matches the orphan bank credit instead of staying UNRESOLVED
    run3 = run_agent_block_once(
        records,
        _make_stub({**base_answers, "UNMATCHED_BANK_UTR_C": _resolution("AMEX_SURCHARGE", 20_000)}),
        tmp_path / "run3.jsonl",
        logic_version=1,
    )
    report = score_agent_runs([run1, run2, run3], ground_truth)
    assert report.honest_exception_consistent is False
    assert report.honest_exception_inconsistent_ids == ["UNMATCHED_BANK_UTR_C"]


# ---------------------------------------------------------------------------
# Test 13: every invocation is logged to data/agent_runs/<seed>_<run_index>.jsonl
# ---------------------------------------------------------------------------

def test_agent_block_every_invocation_logged(tmp_path):
    records = _tiny_records()
    log_path = tmp_path / "999_1.jsonl"
    stub = _make_stub(
        {
            "ORD_A": _resolution("AMEX_SURCHARGE", 1000),
            "ORD_B": _resolution("MISSING_GST", 200),
            "UNMATCHED_BANK_UTR_C": _resolution("UNRESOLVED", 0),
        }
    )
    run_agent_block_once(records, stub, log_path, logic_version=7)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(records) == 3
    for line in lines:
        entry = json.loads(line)
        assert entry["logic_version"] == 7
        assert "record" in entry and "resolution" in entry


# ---------------------------------------------------------------------------
# run_agent_block_resuming (added 2026-09-04): resumes a single run
# interrupted by an external quota wall (AgentRateLimitedError) without
# re-paying for records that run already diagnosed and logged. Real
# occurrence: run 2 of the frozen dataset's 3-run sweep hit exactly this,
# 22/37 records in.
# ---------------------------------------------------------------------------

def test_run_agent_block_resuming_skips_already_logged_records(tmp_path):
    records = _tiny_records()
    log_path = tmp_path / "resume_1.jsonl"

    # Pre-populate the log as if a prior, interrupted attempt already
    # diagnosed ORD_A -- the resume must not call the model for it again.
    calls: list[str] = []

    def _tracking_stub(record):
        calls.append(gt_key_for_record(record))
        answers = {
            "ORD_A": _resolution("AMEX_SURCHARGE", 1000),
            "ORD_B": _resolution("MISSING_GST", 200),
            "UNMATCHED_BANK_UTR_C": _resolution("UNRESOLVED", 0),
        }
        return answers[gt_key_for_record(record)], {"hop_count": 1, "tokens_used": 10}

    run_agent_block_once([records[0]], _tracking_stub, log_path, logic_version=1)  # simulates the prior partial run
    calls.clear()  # only care about calls made during the resume itself

    resolutions = run_agent_block_resuming(records, _tracking_stub, log_path, logic_version=1)

    assert set(calls) == {"ORD_B", "UNMATCHED_BANK_UTR_C"}, "ORD_A was already logged and must not be re-diagnosed"
    assert set(resolutions.keys()) == {"ORD_A", "ORD_B", "UNMATCHED_BANK_UTR_C"}, "the full set must still come back"


def test_run_agent_block_resuming_makes_zero_calls_once_complete(tmp_path):
    records = _tiny_records()
    log_path = tmp_path / "resume_2.jsonl"
    stub = _make_stub(
        {
            "ORD_A": _resolution("AMEX_SURCHARGE", 1000),
            "ORD_B": _resolution("MISSING_GST", 200),
            "UNMATCHED_BANK_UTR_C": _resolution("UNRESOLVED", 0),
        }
    )
    run_agent_block_once(records, stub, log_path, logic_version=1)  # a fully completed prior run

    def _fail_if_called(record):
        raise AssertionError("a fully-logged run must make zero live calls on resume")

    resolutions = run_agent_block_resuming(records, _fail_if_called, log_path, logic_version=1)
    assert len(resolutions) == 3


def test_run_agent_block_resuming_never_reads_a_different_run_indexs_log(tmp_path):
    """The independence guarantee (run_agent_block_once's own docstring:
    never replay another run's answer) must survive resuming -- a resumed
    run only ever reads/writes its OWN log_path."""
    records = _tiny_records()
    other_run_log = tmp_path / "other_run.jsonl"
    this_run_log = tmp_path / "resume_3.jsonl"
    stub = _make_stub(
        {
            "ORD_A": _resolution("AMEX_SURCHARGE", 1000),
            "ORD_B": _resolution("MISSING_GST", 200),
            "UNMATCHED_BANK_UTR_C": _resolution("UNRESOLVED", 0),
        }
    )
    run_agent_block_once(records, stub, other_run_log, logic_version=1)
    assert not this_run_log.exists()

    calls: list[str] = []

    def _tracking_stub(record):
        calls.append(gt_key_for_record(record))
        return stub(record)

    run_agent_block_resuming(records, _tracking_stub, this_run_log, logic_version=1)
    assert set(calls) == {"ORD_A", "ORD_B", "UNMATCHED_BANK_UTR_C"}, "must diagnose all 3 live, ignoring other_run_log entirely"


# ---------------------------------------------------------------------------
# Test 14: the two blocks never merge into one report
# ---------------------------------------------------------------------------

def test_blocks_stay_visually_and_structurally_separate():
    det_report = DeterministicReport(
        batch_run_id=uuid.uuid4(),
        total_orders=87,
        total_unmatched_bank_lines=13,
        fast_path_count=63,
        agent_resolved_count=20,
        honest_exception_count=17,
        false_auto_resolutions=0,
        false_auto_resolution_ids=[],
        adversarial_traps_caught=5,
        adversarial_traps_total=5,
        duplicate_credits_caught=2,
        duplicate_credits_total=2,
        ledger_balance_pass=True,
        trial_balance_rows=[{"account_code": "TOTAL", "account_name": "", "account_type": "", "debit_total_paise": 0, "credit_total_paise": 0, "net_balance_paise": 0}],
        paise_round_trip_drift=0,
    )
    agent_report = AgentBlockReport(
        n_runs=3,
        agent_resolved_denominator=20,
        resolved_per_run=[18, 19, 20],
        correct_root_cause_per_run=[17, 18, 19],
        correct_delta_per_run=[17, 18, 19],
        honest_exception_consistent=True,
        honest_exception_inconsistent_ids=[],
    )

    det_text = format_deterministic_report(det_report)
    agent_text = format_agent_block_report(agent_report)

    assert "Deterministic block" in det_text
    assert "Agent block" in agent_text
    assert "Deterministic block" not in agent_text
    assert "Agent block" not in det_text
    # the deterministic block's exact numbers never leak into the agent
    # block's range report, and vice versa (CLAUDE.md Sec.5)
    assert "[18, 19, 20]" not in det_text
    assert "63" not in agent_text


# ---------------------------------------------------------------------------
# Tests 15-16: --replay re-scores from a log with zero live calls
# ---------------------------------------------------------------------------

def test_replay_reproduces_scores_with_zero_live_calls(tmp_path):
    records = _tiny_records()
    ground_truth = _tiny_ground_truth()
    stub = _make_stub(
        {
            "ORD_A": _resolution("AMEX_SURCHARGE", 1000),
            "ORD_B": _resolution("MISSING_GST", 200),
            "UNMATCHED_BANK_UTR_C": _resolution("UNRESOLVED", 0),
        }
    )
    log_path = tmp_path / "777_1.jsonl"
    live_resolutions = run_agent_block_once(records, stub, log_path, logic_version=1)
    live_report = score_agent_runs([live_resolutions, live_resolutions, live_resolutions], ground_truth)

    def _fail(record):
        raise AssertionError("replay must never call a live diagnose function")

    replayed_resolutions = resolutions_from_log(log_path)
    replayed_report = score_agent_runs([replayed_resolutions, replayed_resolutions, replayed_resolutions], ground_truth)

    assert replayed_report == live_report


def test_replay_errors_clearly_on_missing_record(tmp_path):
    records = _tiny_records()
    incomplete_records = records[:-1]  # drop the UNMATCHED_BANK_UTR_C record
    ground_truth = _tiny_ground_truth()
    stub = _make_stub(
        {
            "ORD_A": _resolution("AMEX_SURCHARGE", 1000),
            "ORD_B": _resolution("MISSING_GST", 200),
        }
    )
    log_path = tmp_path / "888_1.jsonl"
    run_agent_block_once(incomplete_records, stub, log_path, logic_version=1)

    resolutions = resolutions_from_log(log_path)
    with pytest.raises(ValueError, match="missing"):
        score_agent_runs([resolutions, resolutions, resolutions], ground_truth)


# ---------------------------------------------------------------------------
# delta_within_tolerance: exact-at-zero, 1% relative, 5-paise floor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "actual,expected,within",
    [
        (0, 0, True),
        (1, 0, False),
        (-1, 0, False),
        (1000, 1000, True),
        (1005, 1000, True),  # within 1% (10 paise) of 1000
        (1020, 1000, False),  # outside 1%
        (53, 50, True),  # 1% of 50 rounds to 1, but the 5-paise floor covers it
        (55, 50, True),  # exactly at the 5-paise floor
        (56, 50, False),  # just outside the floor
    ],
)
def test_delta_within_tolerance(actual, expected, within):
    assert delta_within_tolerance(actual, expected) is within
