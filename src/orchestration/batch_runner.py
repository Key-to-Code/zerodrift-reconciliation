"""Layer 6 orchestration: sequences Layer 2 (fast path), Layer 4 (agent
diagnosis), and Layer 3 (ledger posting) for one batch, and writes
`reconciliation_matches` rows -- a table the schema/ORM has carried since
Layer 3 but that nothing has ever populated until now.

This module exists because src/api/main.py's own spec (docs/plan.md Layer 6:
"thin transport layer... no business logic here that isn't already in the
layers above") is only true if the per-category posting decision lives
somewhere else. Every posting call below dispatches to an existing, already
-tested src/ledger/journal.py function -- no new tax/matching/timing
arithmetic is added here, only the decision of WHICH existing function to
call for a given discrepancy outcome. Flagged as a new module (outside
docs/plan.md's original file tree) and approved before being written; see
the Layer 6 planning conversation.

Posting design, per discrepancy outcome (approved before implementation):
- fast_path (clean_match / utr_batch): post_clean_match_settlement /
  post_utr_batch_settlement, unchanged from the Layer 3/5 test pattern.
- agent_resolved with a real root cause (AMEX_SURCHARGE, INTL_MARKUP,
  MISSING_GST, MISSING_TDS, CUTOFF_T1, CUTOFF_T2): posted exactly like a
  clean_match, using the settlement's REAL figures. The agent's diagnosis is
  metadata (why it deviates), not a different cash flow -- the money that
  actually moved doesn't change based on why it deviated from the standard
  rate/timing table.
- agent_resolved REFUND_NO_MDR_REVERSAL (refund_clawback): the same
  clean-match posting, plus post_refund_clawback_reversal on top (Layer 3's
  existing tested path).
- honest_exception where the order/settlement DOES exist (short_settlement,
  duplicate_credit, or any other UNRESOLVED settled discrepancy): Stage 1 was
  already posted; post_honest_exception clears AR_GATEWAY_CLEARING straight
  to SUSPENSE_UNRESOLVED for the gross amount. No MDR/GST/TDS lines are
  recognized -- none of the settlement's claimed fee/tax figures are backed
  by a confirmed bank credit, so nothing about them is booked as fact.
- honest_exception where no order exists at all (orphan, adversarial_trap):
  post_honest_exception debits CASH (real money that arrived) against
  SUSPENSE_UNRESOLVED, reference_id UNMATCHED_BANK_<utr> (Layer 1 Addendum
  A4's reserved synthetic identifier).

as_of-gated Stage 2 posting (Layer 6 addendum, approved after Layer 7):
run_batch() originally posted Stage 1 AND Stage 2 for every order
unconditionally, in one synchronous call. That made src/forecast/cashflow.py's
"projected" status (Layer 5) structurally unreachable through any real
triggered run: project_cashflow()'s as_of parameter exists specifically to
model "not everything has settled yet as of this date," but the orchestrator
never respected settlement timing at all, so every posted order was always
already confirmed. This is a genuine Layer 5/6 integration gap, not a Layer 7
concern -- flagged and approved before implementation (see the Layer 7
follow-up conversation).

Fix: an optional `as_of` cutoff. Stage 1 (capture) is still posted
unconditionally for every order -- that reflects real capture timing,
independent of settlement. But Stage 2 (and the corresponding
reconciliation_matches row) is only posted for a settlement/bank-credit whose
own date (settlement_date, or value_date for an unmatched bank line) has
already occurred by `as_of`. An order whose settlement postdates `as_of` is
left exactly where a genuinely partial batch would leave it: captured, not
yet reconciled -- picked up by a later run once `as_of` (or an unspecified
run) advances past it. `as_of=None` (the default) disables the cutoff
entirely and reproduces the original, unconditional behavior byte-for-byte --
every existing test and the frozen dataset's full pipeline are unaffected.

Pre-flight agent-budget check (addendum, 2026-09-03 -- disclosed here, same
pattern as the as_of fix): a live debugging session found that hitting
Groq's daily token cap partway through a seed batch's live agent phase
crashed with some records already posted (Stage 1 capture, fast-path Stage
2, and any discrepancy record diagnosed before the one that hit the wall).
Unlike the frozen path, a seed batch has no cache to fall back on for a
record it has never seen, so there is no soft landing once a live call
becomes necessary. Fix: before ANY posting begins, both discrepancy queues
are built (pure computation) and checked against src/agent/rate_limiter.py's
daily_token_tracker, counting only records that would (a) actually reach
diagnose_fn (an as_of-gated record never does) and (b) actually need a live
call (a cache hit, e.g. the frozen dataset's, needs none) -- multiplied by
the REAL measured average tokens/record from data/agent_runs/ (never a
hardcoded guess). If that estimate exceeds the remaining daily budget,
run_batch raises AgentRateLimitedError immediately, before Stage 1 posting
even starts, so a batch that won't fit fails with zero rows written rather
than partway through. This only applies when diagnose_fn is left at its
default (None) -- an injected diagnose_fn (tests, or a future alternate
backend) is opaque to this estimate and is never gated by it.
"""
from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path
from typing import Callable

from pydantic import BaseModel
from sqlalchemy.orm import Session

from src.agent.discrepancy import (
    DiscrepancyRecord,
    build_settlement_discrepancy_queue,
    build_unmatched_bank_line_queue,
)
from src.agent.resolution import AgentResolution
from src.agent.run_log import diagnose_or_replay
from src.common.money import to_paise
from src.data.models import BankStatementLine, GatewaySettlement, InternalOrder
from src.ledger.journal import (
    AR_GATEWAY_CLEARING,
    CASH,
    post_clean_match_settlement,
    post_honest_exception,
    post_order_capture,
    post_refund_clawback_reversal,
    post_utr_batch_settlement,
)
from src.ledger.models import JournalEntry, ReconciliationMatch
from src.matching.fast_path import ResolvedGroup, run_fast_path
from src.matching.schema import bank_lines_to_frame, orders_to_frame, settlements_to_frame

DEFAULT_AGENT_CACHE_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "agent_runs" / "layer4_test_cache.jsonl"
)

# (resolution, debug_info) -- the same shape src.agent.graph.diagnose_discrepancy
# and src.agent.run_log.diagnose_or_replay already return.
DiagnoseFn = Callable[[DiscrepancyRecord], tuple[AgentResolution, dict]]


def _default_diagnose_fn(agent_cache_path: Path) -> DiagnoseFn:
    from src.agent.graph import AGENT_LOGIC_VERSION

    def _fn(record: DiscrepancyRecord) -> tuple[AgentResolution, dict]:
        resolution, debug_info, _replayed = diagnose_or_replay(record, agent_cache_path, AGENT_LOGIC_VERSION)
        return resolution, debug_info

    return _fn


class BatchRunSummary(BaseModel):
    batch_run_id: uuid.UUID
    total_orders: int
    total_unmatched_bank_lines: int
    fast_path_count: int
    agent_resolved_count: int
    honest_exception_count: int


def _settlement_paise_fields(s: GatewaySettlement) -> dict:
    return {
        "order_id": s.order_id,
        "gross_paise": to_paise(s.gross_amount),
        "mdr_paise": to_paise(s.mdr),
        "gst_paise": to_paise(s.gst_on_mdr),
        "tds_paise": to_paise(s.tds_194o),
        "net_paise": to_paise(s.net_amount),
    }


def _post_resolved_group(
    session: Session, batch_run_id: uuid.UUID, group: ResolvedGroup, settlements_by_order: dict[str, GatewaySettlement]
) -> dict[str, JournalEntry]:
    """Posts Stage 2 (collapsed clean_match, or UTR lump-sum + per-order
    allocation) for a fast-path-resolved group. Returns {order_id: its own
    settlement JournalEntry} so the caller can attach the right
    journal_entry_id to each order's reconciliation_matches row (the
    bank-credit entry for a utr_batch group is shared, not per-order)."""
    if len(group.order_ids) == 1:
        order_id = group.order_ids[0]
        fields = _settlement_paise_fields(settlements_by_order[order_id])
        entry = post_clean_match_settlement(
            session,
            batch_run_id,
            order_id,
            gross_paise=fields["gross_paise"],
            mdr_paise=fields["mdr_paise"],
            gst_paise=fields["gst_paise"],
            tds_paise=fields["tds_paise"],
            net_paise=fields["net_paise"],
        )
        return {order_id: entry}

    orders = [_settlement_paise_fields(settlements_by_order[oid]) for oid in group.order_ids]
    _bank_credit_entry, settlement_entries = post_utr_batch_settlement(
        session, batch_run_id, group.utr, group.net_amount_paise, orders
    )
    return {oid: entry for oid, entry in zip(group.order_ids, settlement_entries)}


def _build_unmatched_bank_line_note(record: DiscrepancyRecord, resolution: AgentResolution) -> str:
    """Threads record.candidate_orders (the near-duplicate order(s) shown to
    the agent as a force-match temptation -- see discrepancy.py's
    find_candidate_orders(), populated only for orphan/adversarial_trap
    records) into the note as candidate_order_id=, so the dashboard's
    Exceptions screen can show the real candidate the agent was given and
    chose not to force-match, instead of nothing. Empty when no candidate
    was close enough in amount/date to be surfaced -- never fabricated.
    Comma-joined for the rare case of 2+ candidates. This is genuinely new
    data on top of the pre-existing 3-field format (discrepancy_reason=;
    root_cause=; <sentence>) -- see api_client.parse_confidence_note_candidate
    for the reader that stays honest about a note built before this change."""
    candidate_ids = ",".join(c.order_id for c in record.candidate_orders)
    return (
        f"discrepancy_reason={record.discrepancy_reason}; root_cause={resolution.root_cause_code}; "
        f"candidate_order_id={candidate_ids}; {resolution.confidence_note}"
    )


def _record_match(
    session: Session,
    batch_run_id: uuid.UUID,
    order_id: str,
    utr: str | None,
    status: str,
    note: str,
    journal_entry_id: int,
    payment_id: str | None = None,
) -> None:
    session.add(
        ReconciliationMatch(
            batch_run_id=batch_run_id,
            order_id=order_id,
            payment_id=payment_id,
            utr=utr,
            status=status,
            confidence_note=note,
            journal_entry_id=journal_entry_id,
        )
    )


def run_batch(
    session: Session,
    batch_run_id: uuid.UUID,
    orders: list[InternalOrder],
    settlements: list[GatewaySettlement],
    bank_lines: list[BankStatementLine],
    diagnose_fn: DiagnoseFn | None = None,
    agent_cache_path: Path = DEFAULT_AGENT_CACHE_PATH,
    as_of: date | None = None,
) -> BatchRunSummary:
    """Runs the full pipeline for one batch and posts every result to the
    ledger, writing one reconciliation_matches row per real order and one
    per unmatched bank line (Layer 1 Addendum A4's UNMATCHED_BANK_<utr>
    synthetic identifier). Assumes batch_run_id is fresh -- not idempotent
    against being called twice with the same id (the API always mints a new
    uuid4() per trigger, per docs/plan.md Layer 6).

    as_of, if given, gates Stage 2: a settlement/bank-credit dated after
    as_of is left at Stage 1 only (captured, not yet reconciled) -- see the
    module docstring's "as_of-gated Stage 2 posting" section. as_of=None
    (the default) posts everything unconditionally, unchanged from the
    original behavior."""
    orders_by_id = {o.order_id: o for o in orders}
    settlements_by_order = {s.order_id: s for s in settlements}

    orders_df = orders_to_frame(orders)
    settlements_df = settlements_to_frame(settlements)
    bank_df = bank_lines_to_frame(bank_lines)

    # Both discrepancy queues are pure computation (no DB writes) -- built
    # here, before ANY posting, specifically so a pre-flight budget check
    # can run with zero partial progress to undo if it fails. Reused below
    # by the actual posting loops instead of being rebuilt inline.
    settlement_discrepancy_records = build_settlement_discrepancy_queue(orders_df, settlements_df, bank_df)
    unmatched_records = build_unmatched_bank_line_queue(orders_df, settlements_df, bank_df)

    if diagnose_fn is None:
        diagnose_fn = _default_diagnose_fn(agent_cache_path)
        # Pre-flight budget check -- only meaningful for the real default
        # cache-backed path (an injected diagnose_fn, e.g. a test stub, is
        # opaque and never touches the real cache/budget anyway). Only
        # records that would actually be diagnosed matter -- an as_of-gated
        # record never reaches diagnose_fn at all, so it must not count
        # toward the estimate (docs/plan.md Layer 6's as_of addendum).
        # Added after a live debugging session found that hitting the daily
        # quota mid-seed-batch crashed with some records already posted --
        # see src/agent/rate_limiter.py::check_budget_for_batch.
        from src.agent.graph import AGENT_LOGIC_VERSION
        from src.agent.rate_limiter import check_budget_for_batch
        from src.agent.run_log import average_real_tokens_per_live_call, count_live_calls_needed

        records_needing_diagnosis = [
            r
            for r in settlement_discrepancy_records
            if not (as_of is not None and settlements_by_order[r.order_context.order_id].settlement_date > as_of)
        ] + [
            r
            for r in unmatched_records
            if not (as_of is not None and date.fromisoformat(r.bank_credits[0].value_date) > as_of)
        ]
        n_live_calls_needed = count_live_calls_needed(records_needing_diagnosis, agent_cache_path, AGENT_LOGIC_VERSION)
        avg_tokens_per_call = average_real_tokens_per_live_call([agent_cache_path])
        check_budget_for_batch(n_live_calls_needed, avg_tokens_per_call)

    for order in orders:
        post_order_capture(session, batch_run_id, order.order_id, to_paise(order.gross_amount))

    fast_path_result = run_fast_path(orders_df, settlements_df, bank_df)

    fast_path_count = agent_resolved_count = honest_exception_count = 0

    for group in fast_path_result.resolved:
        if as_of is not None and any(
            settlements_by_order[oid].settlement_date > as_of for oid in group.order_ids
        ):
            continue  # this settlement genuinely hasn't happened yet as of this clock
        entries_by_order = _post_resolved_group(session, batch_run_id, group, settlements_by_order)
        for order_id in group.order_ids:
            settlement = settlements_by_order[order_id]
            _record_match(
                session,
                batch_run_id,
                order_id=order_id,
                utr=group.utr,
                status="fast_path",
                note=f"fast_path: {group.match_method}",
                journal_entry_id=entries_by_order[order_id].entry_id,
                payment_id=settlement.payment_id,
            )
            fast_path_count += 1

    for record in settlement_discrepancy_records:
        order_id = record.order_context.order_id
        order = orders_by_id[order_id]
        settlement = settlements_by_order[order_id]
        if as_of is not None and settlement.settlement_date > as_of:
            continue  # this settlement genuinely hasn't happened yet as of this clock
        resolution, _debug_info = diagnose_fn(record)
        note = (
            f"discrepancy_reason={record.discrepancy_reason}; root_cause={resolution.root_cause_code}; "
            f"delta_paise={resolution.quantified_delta_paise}; {resolution.confidence_note}"
        )

        if resolution.root_cause_code == "UNRESOLVED":
            entry = post_honest_exception(
                session,
                batch_run_id,
                order_id,
                AR_GATEWAY_CLEARING,
                "C",
                to_paise(settlement.gross_amount),
                note=note,
            )
            status = "honest_exception"
            honest_exception_count += 1
        else:
            fields = _settlement_paise_fields(settlement)
            entry = post_clean_match_settlement(
                session,
                batch_run_id,
                order_id,
                gross_paise=fields["gross_paise"],
                mdr_paise=fields["mdr_paise"],
                gst_paise=fields["gst_paise"],
                tds_paise=fields["tds_paise"],
                net_paise=fields["net_paise"],
            )
            if resolution.root_cause_code == "REFUND_NO_MDR_REVERSAL":
                if order.refund_amount is None:
                    raise ValueError(
                        f"{order_id}: agent returned REFUND_NO_MDR_REVERSAL but the order carries no refund_amount"
                    )
                entry = post_refund_clawback_reversal(
                    session, batch_run_id, order_id, to_paise(order.refund_amount)
                )
            status = "agent_resolved"
            agent_resolved_count += 1

        _record_match(
            session,
            batch_run_id,
            order_id=order_id,
            utr=settlement.utr,
            status=status,
            note=note,
            journal_entry_id=entry.entry_id,
            payment_id=settlement.payment_id,
        )

    for record in unmatched_records:
        bank_credit = record.bank_credits[0]
        if as_of is not None and date.fromisoformat(bank_credit.value_date) > as_of:
            continue  # this bank credit genuinely hasn't happened yet as of this clock
        resolution, _debug_info = diagnose_fn(record)
        note = _build_unmatched_bank_line_note(record, resolution)
        reference_id = f"UNMATCHED_BANK_{bank_credit.utr}"
        entry = post_honest_exception(
            session, batch_run_id, reference_id, CASH, "D", bank_credit.credited_amount_paise, note=note
        )
        _record_match(
            session,
            batch_run_id,
            order_id=reference_id,
            utr=bank_credit.utr,
            status="honest_exception",
            note=note,
            journal_entry_id=entry.entry_id,
        )
        honest_exception_count += 1

    session.commit()

    return BatchRunSummary(
        batch_run_id=batch_run_id,
        total_orders=len(orders),
        total_unmatched_bank_lines=len(unmatched_records),
        fast_path_count=fast_path_count,
        agent_resolved_count=agent_resolved_count,
        honest_exception_count=honest_exception_count,
    )
