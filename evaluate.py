"""Layer 8 evaluation harness (docs/plan.md Layer 8).

Two blocks, deliberately never merged into one figure (CLAUDE.md Sec.5):

- The DETERMINISTIC block runs the full pipeline (fast path -> agent ->
  ledger) once against a batch, via src.orchestration.batch_runner.run_batch,
  and reports exact counts plus the full trial balance. Against the frozen
  challenge_batch_100 dataset this makes zero live model calls -- every
  discrepancy-queue record is already covered by
  data/agent_runs/layer4_test_cache.jsonl (src/agent/run_log.py), so the
  numbers are byte-reproducible run to run.
- The AGENT block runs the live diagnostic agent independently N times
  (default 3) over every discrepancy-queue record -- never replayed from
  cache, since the whole point is genuine run-to-run variance -- and reports
  a min/median/max range, never a single number. Every invocation is logged
  to data/agent_runs/<seed>_<run_index>.jsonl; `--replay <prefix>` re-scores
  those logs with zero live calls (the offline demo fallback).

Layer 8 scope note (approved during planning): this module builds and tests
the scoring/variance/logging/replay machinery using an injectable diagnose_fn
(the same test-isolation seam src/api/main.py's get_diagnose_fn and
tests/test_api.py's stub already use) -- it does not itself spend live model
budget running the real 3x sweep. That real sweep, and writing its resulting
numbers into the README, is a deliberate Layer 9 action a human runs when
ready (see docs/plan.md Layer 8's methodology note on Groq's daily token
cap), never something pytest does on every run.
"""
from __future__ import annotations

import argparse
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Callable

from sqlalchemy import select
from sqlalchemy.orm import Session

from src.agent.discrepancy import (
    DiscrepancyRecord,
    build_settlement_discrepancy_queue,
    build_unmatched_bank_line_queue,
)
from src.agent.resolution import AgentResolution
from src.agent.run_log import append_run_log, load_run_log
from src.common.money import from_paise, to_paise
from src.data.generator import generate_batch
from src.data.models import BankStatementLine, GatewaySettlement, GroundTruthEntry, InternalOrder
from src.ledger.journal import trial_balance
from src.ledger.models import ReconciliationMatch, ensure_schema_exists, get_engine, get_sessionmaker
from src.matching.schema import bank_lines_to_frame, orders_to_frame, settlements_to_frame
from src.orchestration.batch_runner import DiagnoseFn, run_batch

PROJECT_ROOT = Path(__file__).resolve().parent
FROZEN_DIR = PROJECT_ROOT / "data" / "challenge_batch_100"
AGENT_RUNS_DIR = PROJECT_ROOT / "data" / "agent_runs"

# expected_delta_paise tolerance: exact match required at zero (a percentage
# tolerance is undefined there -- cutoff_drift's expected_delta_paise is 0
# for all 5 ground_truth.json records); otherwise max(1% relative, +/-5
# paise floor). The floor exists because 1% of a small nonzero expected
# delta (e.g. 50 paise) rounds to 0, which would silently collapse back to
# an exact-match requirement nobody intended.
DELTA_TOLERANCE_RELATIVE = 0.01
DELTA_TOLERANCE_FLOOR_PAISE = 5


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_frozen_dataset() -> tuple[
    list[InternalOrder], list[GatewaySettlement], list[BankStatementLine], list[GroundTruthEntry]
]:
    import json

    orders = [
        InternalOrder.model_validate(d)
        for d in json.loads((FROZEN_DIR / "internal_orders.json").read_text(encoding="utf-8"))
    ]
    settlements = [
        GatewaySettlement.model_validate(d)
        for d in json.loads((FROZEN_DIR / "gateway_settlement.json").read_text(encoding="utf-8"))
    ]
    bank_lines = [
        BankStatementLine.model_validate(d)
        for d in json.loads((FROZEN_DIR / "bank_statement.json").read_text(encoding="utf-8"))
    ]
    ground_truth = [
        GroundTruthEntry.model_validate(d)
        for d in json.loads((FROZEN_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    ]
    return orders, settlements, bank_lines, ground_truth


def load_seed_dataset(
    seed: int, records: int
) -> tuple[list[InternalOrder], list[GatewaySettlement], list[BankStatementLine], list[GroundTruthEntry]]:
    batch = generate_batch(num_records=records, seed=seed)
    return batch.orders, batch.settlements, batch.bank_lines, batch.ground_truth


def ground_truth_lookup(ground_truth: list[GroundTruthEntry]) -> dict[str, GroundTruthEntry]:
    return {g.order_id: g for g in ground_truth}


def gt_key_for_record(record: DiscrepancyRecord) -> str:
    """ground_truth.json's own order_id convention for a DiscrepancyRecord:
    the real order_id for a settlement-side record, or the
    UNMATCHED_BANK_<utr> synthetic id (Layer 1 Addendum A4) for an
    unmatched-bank-line record. Deliberately NOT the same as
    src.agent.run_log.record_key(), which uses an "unmatched:<utr>" scheme
    for cache-file identity only -- that keying has nothing to do with
    scoring against ground truth."""
    if record.order_context is not None:
        return record.order_context.order_id
    return f"UNMATCHED_BANK_{record.bank_credits[0].utr}"


def build_discrepancy_queue(
    orders: list[InternalOrder], settlements: list[GatewaySettlement], bank_lines: list[BankStatementLine]
) -> list[DiscrepancyRecord]:
    orders_df = orders_to_frame(orders)
    settlements_df = settlements_to_frame(settlements)
    bank_df = bank_lines_to_frame(bank_lines)
    return build_settlement_discrepancy_queue(orders_df, settlements_df, bank_df) + build_unmatched_bank_line_queue(
        orders_df, settlements_df, bank_df
    )


def paise_round_trip_drift(
    orders: list[InternalOrder], settlements: list[GatewaySettlement], bank_lines: list[BankStatementLine]
) -> int:
    """Number of money fields across the whole batch where
    from_paise(to_paise(x)) != x. Must be 0 -- this is CLAUDE.md Sec.3's
    reproducibility guarantee, measured directly rather than merely
    asserted."""
    drift = 0
    for order in orders:
        values = [order.gross_amount]
        if order.refund_amount is not None:
            values.append(order.refund_amount)
        for value in values:
            if from_paise(to_paise(value)) != value:
                drift += 1
    for settlement in settlements:
        for value in (
            settlement.gross_amount,
            settlement.mdr,
            settlement.gst_on_mdr,
            settlement.tds_194o,
            settlement.net_amount,
        ):
            if from_paise(to_paise(value)) != value:
                drift += 1
    for bank_line in bank_lines:
        if from_paise(to_paise(bank_line.credited_amount)) != bank_line.credited_amount:
            drift += 1
    return drift


# ---------------------------------------------------------------------------
# Deterministic block
# ---------------------------------------------------------------------------

@dataclass
class DeterministicReport:
    batch_run_id: uuid.UUID
    total_orders: int
    total_unmatched_bank_lines: int
    fast_path_count: int
    agent_resolved_count: int
    honest_exception_count: int
    false_auto_resolutions: int
    false_auto_resolution_ids: list[str]
    adversarial_traps_caught: int
    adversarial_traps_total: int
    duplicate_credits_caught: int
    duplicate_credits_total: int
    ledger_balance_pass: bool
    trial_balance_rows: list[dict]
    paise_round_trip_drift: int


def run_deterministic_block(
    session: Session,
    batch_run_id: uuid.UUID,
    orders: list[InternalOrder],
    settlements: list[GatewaySettlement],
    bank_lines: list[BankStatementLine],
    ground_truth: list[GroundTruthEntry],
    diagnose_fn: DiagnoseFn | None = None,
) -> DeterministicReport:
    summary = run_batch(session, batch_run_id, orders, settlements, bank_lines, diagnose_fn=diagnose_fn)
    gt_by_id = ground_truth_lookup(ground_truth)

    matches = (
        session.execute(select(ReconciliationMatch).where(ReconciliationMatch.batch_run_id == batch_run_id))
        .scalars()
        .all()
    )

    false_auto_ids: list[str] = []
    adversarial_total = sum(1 for g in ground_truth if g.category == "adversarial_trap")
    duplicate_total = sum(1 for g in ground_truth if g.category == "duplicate_credit")
    adversarial_caught = 0
    duplicate_caught = 0
    for match in matches:
        gt = gt_by_id.get(match.order_id)
        if gt is None:
            raise ValueError(f"reconciliation_matches row {match.order_id!r} has no ground_truth.json entry")
        if match.status in ("fast_path", "agent_resolved") and gt.expected_resolution == "honest_exception":
            false_auto_ids.append(match.order_id)
        if gt.category == "adversarial_trap" and match.status == "honest_exception":
            adversarial_caught += 1
        if gt.category == "duplicate_credit" and match.status == "honest_exception":
            duplicate_caught += 1

    tb_rows = trial_balance(session, batch_run_id).to_dicts()
    total_row = next(row for row in tb_rows if row["account_code"] == "TOTAL")

    return DeterministicReport(
        batch_run_id=batch_run_id,
        total_orders=summary.total_orders,
        total_unmatched_bank_lines=summary.total_unmatched_bank_lines,
        fast_path_count=summary.fast_path_count,
        agent_resolved_count=summary.agent_resolved_count,
        honest_exception_count=summary.honest_exception_count,
        false_auto_resolutions=len(false_auto_ids),
        false_auto_resolution_ids=false_auto_ids,
        adversarial_traps_caught=adversarial_caught,
        adversarial_traps_total=adversarial_total,
        duplicate_credits_caught=duplicate_caught,
        duplicate_credits_total=duplicate_total,
        ledger_balance_pass=(total_row["net_balance_paise"] == 0),
        trial_balance_rows=tb_rows,
        paise_round_trip_drift=paise_round_trip_drift(orders, settlements, bank_lines),
    )


def format_deterministic_report(report: DeterministicReport) -> str:
    total_records = report.total_orders + report.total_unmatched_bank_lines
    lines = [
        "=== Deterministic block (exact, byte-reproducible for this batch) ===",
        f"Fast path resolved:       {report.fast_path_count} / {total_records}",
        f"Agent resolved:           {report.agent_resolved_count} / {total_records}",
        f"Honest exceptions:        {report.honest_exception_count} / {total_records}",
        f"False auto-resolutions:   {report.false_auto_resolutions}   (must be 0)",
        f"Adversarial traps caught: {report.adversarial_traps_caught} / {report.adversarial_traps_total}"
        "   (must be 5/5)",
        f"Duplicate credits caught: {report.duplicate_credits_caught} / {report.duplicate_credits_total}"
        "   (must be 2/2)",
        f"Ledger balance check:     {'PASS' if report.ledger_balance_pass else 'FAIL'}",
        f"Paise round-trip drift:   {report.paise_round_trip_drift}   (asserted 0)",
        "",
        "Trial balance (paise):",
    ]
    for row in report.trial_balance_rows:
        lines.append(
            f"  {row['account_code']:<24} debit={row['debit_total_paise']:>14,} "
            f"credit={row['credit_total_paise']:>14,} net={row['net_balance_paise']:>14,}"
        )
    if report.false_auto_resolution_ids:
        lines.append(f"False auto-resolution order_ids: {report.false_auto_resolution_ids}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent block
# ---------------------------------------------------------------------------

def delta_within_tolerance(actual_paise: int, expected_paise: int) -> bool:
    if expected_paise == 0:
        return actual_paise == 0
    tolerance = max(round(abs(expected_paise) * DELTA_TOLERANCE_RELATIVE), DELTA_TOLERANCE_FLOOR_PAISE)
    return abs(actual_paise - expected_paise) <= tolerance


def run_agent_block_once(
    records: list[DiscrepancyRecord], diagnose_fn: DiagnoseFn, log_path: Path, logic_version: int
) -> dict[str, AgentResolution]:
    """One independent sweep over every discrepancy-queue record via
    diagnose_fn -- never cached/replayed internally, since the agent block's
    entire purpose is measuring genuine run-to-run variance. Every
    invocation is appended to log_path regardless of source, so --replay can
    reconstruct this exact run later with zero live calls. Returns
    resolutions keyed by gt_key_for_record (ground_truth.json's order_id
    convention), not run_log.record_key's separate cache-identity scheme."""
    resolutions: dict[str, AgentResolution] = {}
    for record in records:
        resolution, debug_info = diagnose_fn(record)
        append_run_log(log_path, record, resolution, debug_info, logic_version)
        resolutions[gt_key_for_record(record)] = resolution
    return resolutions


def resolutions_from_log(log_path: Path) -> dict[str, AgentResolution]:
    """Reconstructs a run's resolutions, keyed by gt_key_for_record, from a
    previously-written data/agent_runs/<seed>_<run_index>.jsonl log -- zero
    live calls."""
    entries = load_run_log(log_path)
    resolutions: dict[str, AgentResolution] = {}
    for entry in entries.values():
        record = DiscrepancyRecord.model_validate(entry["record"])
        resolutions[gt_key_for_record(record)] = AgentResolution.model_validate(entry["resolution"])
    return resolutions


@dataclass
class AgentBlockReport:
    n_runs: int
    agent_resolved_denominator: int
    resolved_per_run: list[int]
    correct_root_cause_per_run: list[int]
    correct_delta_per_run: list[int]
    honest_exception_consistent: bool
    honest_exception_inconsistent_ids: list[str] = field(default_factory=list)


def score_agent_runs(
    run_resolutions: list[dict[str, AgentResolution]], ground_truth: list[GroundTruthEntry]
) -> AgentBlockReport:
    """Scores N independent agent-block runs against ground_truth.json.
    agent_resolved_denominator is derived from the actual ground_truth
    passed in (20 for the frozen challenge_batch_100 dataset: 7 fee_drift +
    5 missing_tax_line + 5 cutoff_drift + 3 refund_clawback) rather than
    hardcoded, since a live --seed run can use a differently-sized batch
    (docs/plan.md Layer 1: "scale proportionally for other sizes").

    Raises ValueError if any run is missing a resolution for a record
    ground_truth.json expects one for -- this applies equally to a live run
    (a real bug) and to --replay (an incomplete/mismatched log), since
    silently skipping a missing record would produce a quietly-wrong score
    either way."""
    agent_resolved_gt = [g for g in ground_truth if g.expected_resolution == "agent_resolved"]
    honest_exception_gt = [g for g in ground_truth if g.expected_resolution == "honest_exception"]

    expected_keys = [g.order_id for g in agent_resolved_gt] + [g.order_id for g in honest_exception_gt]
    for i, resolutions in enumerate(run_resolutions):
        missing = [key for key in expected_keys if key not in resolutions]
        if missing:
            raise ValueError(f"run {i}: missing resolutions for {missing} -- cannot score incompletely")

    resolved_per_run: list[int] = []
    correct_root_per_run: list[int] = []
    correct_delta_per_run: list[int] = []
    for resolutions in run_resolutions:
        resolved = correct_root = correct_delta = 0
        for gt in agent_resolved_gt:
            resolution = resolutions[gt.order_id]
            if resolution.root_cause_code != "UNRESOLVED":
                resolved += 1
            if resolution.root_cause_code == gt.expected_root_cause_code:
                correct_root += 1
            if delta_within_tolerance(resolution.quantified_delta_paise, gt.expected_delta_paise):
                correct_delta += 1
        resolved_per_run.append(resolved)
        correct_root_per_run.append(correct_root)
        correct_delta_per_run.append(correct_delta)

    inconsistent_ids = [
        gt.order_id
        for gt in honest_exception_gt
        if {resolutions[gt.order_id].root_cause_code for resolutions in run_resolutions} != {"UNRESOLVED"}
    ]

    return AgentBlockReport(
        n_runs=len(run_resolutions),
        agent_resolved_denominator=len(agent_resolved_gt),
        resolved_per_run=resolved_per_run,
        correct_root_cause_per_run=correct_root_per_run,
        correct_delta_per_run=correct_delta_per_run,
        honest_exception_consistent=(len(inconsistent_ids) == 0),
        honest_exception_inconsistent_ids=inconsistent_ids,
    )


def format_agent_block_report(report: AgentBlockReport) -> str:
    denom = report.agent_resolved_denominator
    lines = [
        f"=== Agent block ({report.n_runs} independent live runs -- range, never a single number) ===",
        f"Agent resolved (per run):           {report.resolved_per_run}  out of {denom}",
        f"  min / median / max:               "
        f"{min(report.resolved_per_run)} / {median(report.resolved_per_run)} / {max(report.resolved_per_run)}",
        f"Correct root_cause_code (per run):  {report.correct_root_cause_per_run}  out of {denom}",
        f"Correct quantified_delta (per run): {report.correct_delta_per_run}  out of {denom}",
        f"Honest exceptions from agent path: consistent across all {report.n_runs} runs? "
        f"{'Y' if report.honest_exception_consistent else 'N'}",
    ]
    if not report.honest_exception_consistent:
        lines.append(f"  inconsistent order_ids: {report.honest_exception_inconsistent_ids}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _replay_prefix_report(prefix: str, n_runs: int, ground_truth: list[GroundTruthEntry]) -> AgentBlockReport:
    run_resolutions = [resolutions_from_log(Path(f"{prefix}_{i}.jsonl")) for i in range(1, n_runs + 1)]
    return score_agent_runs(run_resolutions, ground_truth)


def main() -> None:
    parser = argparse.ArgumentParser(description="Layer 8 evaluation harness for ZeroDrift.")
    parser.add_argument("--seed", type=int, default=None, help="Live unseen-seed batch instead of the frozen one.")
    parser.add_argument("--records", type=int, default=100)
    parser.add_argument("--runs", type=int, default=3, help="Number of independent live agent-block sweeps.")
    parser.add_argument(
        "--replay",
        type=str,
        default=None,
        help="Prefix of previously-logged agent runs to re-score offline, e.g. data/agent_runs/42 "
        "for 42_1.jsonl/42_2.jsonl/42_3.jsonl. Skips the deterministic block and every live call.",
    )
    parser.add_argument(
        "--skip-agent-block", action="store_true", help="Only run the deterministic block (no live agent calls)."
    )
    args = parser.parse_args()

    if args.seed is None:
        orders, settlements, bank_lines, ground_truth = load_frozen_dataset()
    else:
        orders, settlements, bank_lines, ground_truth = load_seed_dataset(args.seed, args.records)

    if args.replay:
        report = _replay_prefix_report(args.replay, args.runs, ground_truth)
        print(format_agent_block_report(report))
        return

    engine = get_engine()
    ensure_schema_exists(engine)
    session = get_sessionmaker(engine)()
    try:
        det_report = run_deterministic_block(
            session, uuid.uuid4(), orders, settlements, bank_lines, ground_truth
        )
    finally:
        session.close()
    print(format_deterministic_report(det_report))

    if args.skip_agent_block:
        return

    from src.agent.graph import AGENT_LOGIC_VERSION, diagnose_discrepancy

    records = build_discrepancy_queue(orders, settlements, bank_lines)
    seed_label = args.seed if args.seed is not None else "frozen"
    run_resolutions = []
    for i in range(1, args.runs + 1):
        log_path = AGENT_RUNS_DIR / f"{seed_label}_{i}.jsonl"
        run_resolutions.append(run_agent_block_once(records, diagnose_discrepancy, log_path, AGENT_LOGIC_VERSION))
    print()
    print(format_agent_block_report(score_agent_runs(run_resolutions, ground_truth)))


if __name__ == "__main__":
    main()
