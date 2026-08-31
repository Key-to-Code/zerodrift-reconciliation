"""Layer 4 tests for the bounded diagnostic agent
(src/agent/resolution.py, src/agent/discrepancy.py, src/agent/graph.py).
Written before implementation per CLAUDE.md's build protocol.

Two kinds of test live in this file, and it matters which is which:

- LIVE tests call diagnose_or_replay() (src/agent/run_log.py), which calls
  the real Groq-hosted openai/gpt-oss-120b model (see src/agent/graph.py's
  module docstring for why Groq/this model, not Claude, in dev) ONLY if no
  cached result exists for that record under the current AGENT_LOGIC_VERSION
  -- otherwise it replays the cached result with zero API calls. This exists
  because Groq's free tier caps at 200,000 tokens/day (~4,400 tokens per
  record-run observed here -> ~45 record-runs/day), which a blind full-suite
  live re-run every time would exhaust quickly. These are still the tests
  that prove something about the agent's actual behavior, on whichever pass
  actually calls the model live.
- DETERMINISTIC tests call _invoke_tool_logic()/_propose_resolution_logic()
  directly with an injected stub model object. They never call
  diagnose_or_replay(), _build_model(), or .bind_tools(), and never touch
  the network or the cache -- see graph.py's module docstring on the
  testability seam. These prove things about this repo's own orchestration
  code (the 3-call cap, the malformed-payload retry/fallback, the
  pseudo-tool error safety net), not about model behavior, and are fast and
  free to run as often as needed.

The adversarial_trap tests (1, 2) were de-risked by a manual spike before
this file was written: the original SYSTEM_PROMPT design described the
final-answer JSON schema in the same turn as bound tools, which caused
openai/gpt-oss-120b to occasionally short-circuit via a call to a synthetic
"json" tool not in the bound tool list -- once accepted, this let one trap
force-match on AMEX_SURCHARGE. Separating the schema instruction out (now
FINAL_ANSWER_INSTRUCTION, shown only in propose_resolution) fixed it; 15/15
across 3 full runs passed afterward. Test 15 is the permanent regression
guard for that exact bug.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import httpx
import pytest
from dotenv import load_dotenv
from groq import BadRequestError
from langchain_core.messages import AIMessage

load_dotenv()

from src.agent import graph
from src.agent.discrepancy import (
    DiscrepancyRecord,
    build_settlement_discrepancy_queue,
    build_unmatched_bank_line_queue,
)
from src.agent.graph import (
    AGENT_LOGIC_VERSION,
    AgentState,
    MAX_TOOL_CALLS,
    _invoke_tool_logic,
    _propose_resolution_logic,
)
from src.agent.run_log import diagnose_or_replay
from src.data.models import BankStatementLine, GatewaySettlement, GroundTruthEntry, InternalOrder
from src.matching.schema import bank_lines_to_frame, orders_to_frame, settlements_to_frame

FROZEN_DIR = Path(__file__).resolve().parents[1] / "data" / "challenge_batch_100"

# Dev/test cache -- see src/agent/run_log.py's module docstring. Once a
# record has been successfully diagnosed against the current
# AGENT_LOGIC_VERSION, later test runs replay it with zero API calls
# instead of re-asking Groq, so a full pytest run doesn't re-spend the
# free-tier daily token budget rediscovering the same answers.
CACHE_PATH = Path(__file__).resolve().parents[1] / "data" / "agent_runs" / "layer4_test_cache.jsonl"


# ---------------------------------------------------------------------------
# Fixtures: the frozen challenge_batch_100 dataset, parsed and framed once,
# plus both discrepancy queues built once (module scope -- these queues are
# pure data, no LLM calls happen building them).
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
def ground_truth_by_order_id(frozen_ground_truth) -> dict[str, GroundTruthEntry]:
    return {e.order_id: e for e in frozen_ground_truth}


@pytest.fixture(scope="module")
def orders_df(frozen_orders):
    return orders_to_frame(frozen_orders)


@pytest.fixture(scope="module")
def settlements_df(frozen_settlements):
    return settlements_to_frame(frozen_settlements)


@pytest.fixture(scope="module")
def bank_df(frozen_bank_lines):
    return bank_lines_to_frame(frozen_bank_lines)


@pytest.fixture(scope="module")
def unmatched_queue(orders_df, settlements_df, bank_df) -> list[DiscrepancyRecord]:
    return build_unmatched_bank_line_queue(orders_df, settlements_df, bank_df)


@pytest.fixture(scope="module")
def settlement_queue(orders_df, settlements_df, bank_df) -> list[DiscrepancyRecord]:
    return build_settlement_discrepancy_queue(orders_df, settlements_df, bank_df)


def _gt_key_for_unmatched(record: DiscrepancyRecord) -> str:
    return f"UNMATCHED_BANK_{record.bank_credits[0].utr}"


def _unmatched_by_category(unmatched_queue, ground_truth_by_order_id, category: str) -> list[DiscrepancyRecord]:
    return [r for r in unmatched_queue if ground_truth_by_order_id[_gt_key_for_unmatched(r)].category == category]


def _settled_by_category(settlement_queue, ground_truth_by_order_id, category: str) -> list[DiscrepancyRecord]:
    return [
        r for r in settlement_queue if ground_truth_by_order_id[r.order_context.order_id].category == category
    ]


def _fresh_state(**overrides) -> AgentState:
    base: AgentState = {
        "record": None,
        "hop_count": 0,
        "max_hops": MAX_TOOL_CALLS,
        "tool_call_history": [],
        "messages": [],
        "resolution": None,
        "status": "PROCESSING",
        "raw_failure": None,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# LIVE tests 1-2 -- adversarial_trap (the single most important test in this
# layer; de-risked by the manual spike before this file was written).
# ---------------------------------------------------------------------------

def test_adversarial_trap_never_force_matched(unmatched_queue, ground_truth_by_order_id):
    """Proves acceptance criterion: all 5 adversarial_trap records reach
    honest_exception via the live agent, never force-matched."""
    traps = _unmatched_by_category(unmatched_queue, ground_truth_by_order_id, "adversarial_trap")
    assert len(traps) == 5
    for record in traps:
        resolution, debug, _replayed = diagnose_or_replay(record, CACHE_PATH, AGENT_LOGIC_VERSION)
        assert resolution.root_cause_code == "UNRESOLVED", (
            f"adversarial_trap {record.bank_credits[0].utr} force-matched: "
            f"{resolution.root_cause_code} delta={resolution.quantified_delta_paise}"
        )
        assert debug["hop_count"] <= MAX_TOOL_CALLS


def test_adversarial_trap_candidate_surfaced(unmatched_queue, ground_truth_by_order_id):
    """Proves the trap test above is testing something real: each trap's
    candidate_orders must contain exactly the ground-truth twin order, not
    be empty (toothless trap) or diluted with noise. Ground truth notes are
    read here for verification only -- never fed to the agent itself."""
    traps = _unmatched_by_category(unmatched_queue, ground_truth_by_order_id, "adversarial_trap")
    assert len(traps) == 5
    for record in traps:
        gt = ground_truth_by_order_id[_gt_key_for_unmatched(record)]
        expected_twin = re.search(r"ORD\d+", gt.notes).group()
        discovered_ids = [c.order_id for c in record.candidate_orders]
        assert discovered_ids == [expected_twin], (
            f"trap {record.bank_credits[0].utr}: expected candidates=[{expected_twin}], got {discovered_ids}"
        )


# ---------------------------------------------------------------------------
# LIVE tests 3-6 -- the correctness-graded categories. root_cause_code AND
# quantified_delta_paise must match ground_truth.json exactly, not merely
# "isn't honest_exception".
# ---------------------------------------------------------------------------

def test_fee_drift_resolved_correctly(settlement_queue, ground_truth_by_order_id):
    records = _settled_by_category(settlement_queue, ground_truth_by_order_id, "fee_drift")
    assert len(records) == 7
    for record in records:
        gt = ground_truth_by_order_id[record.order_context.order_id]
        resolution, debug, _replayed = diagnose_or_replay(record, CACHE_PATH, AGENT_LOGIC_VERSION)
        assert resolution.root_cause_code == gt.expected_root_cause_code, (
            f"{gt.order_id}: expected {gt.expected_root_cause_code}, got {resolution.root_cause_code}"
        )
        assert resolution.quantified_delta_paise == gt.expected_delta_paise, (
            f"{gt.order_id}: expected delta {gt.expected_delta_paise}, got {resolution.quantified_delta_paise}"
        )
        assert debug["hop_count"] <= MAX_TOOL_CALLS


def test_missing_tax_line_resolved_correctly(settlement_queue, ground_truth_by_order_id):
    records = _settled_by_category(settlement_queue, ground_truth_by_order_id, "missing_tax_line")
    assert len(records) == 5
    for record in records:
        gt = ground_truth_by_order_id[record.order_context.order_id]
        resolution, debug, _replayed = diagnose_or_replay(record, CACHE_PATH, AGENT_LOGIC_VERSION)
        assert resolution.root_cause_code == gt.expected_root_cause_code, (
            f"{gt.order_id}: expected {gt.expected_root_cause_code}, got {resolution.root_cause_code}"
        )
        assert resolution.quantified_delta_paise == gt.expected_delta_paise, (
            f"{gt.order_id}: expected delta {gt.expected_delta_paise}, got {resolution.quantified_delta_paise}"
        )
        assert debug["hop_count"] <= MAX_TOOL_CALLS


def test_cutoff_drift_resolved_correctly(settlement_queue, ground_truth_by_order_id):
    """cutoff_drift is a pure timing violation -- confirmed against the real
    frozen dataset that expected_delta_paise is 0 for all 5 records (no
    money component), so this asserts that explicitly rather than assuming
    a nonzero delta the way a copy-pasted correctness test might."""
    records = _settled_by_category(settlement_queue, ground_truth_by_order_id, "cutoff_drift")
    assert len(records) == 5
    for record in records:
        gt = ground_truth_by_order_id[record.order_context.order_id]
        assert gt.expected_delta_paise == 0
        resolution, debug, _replayed = diagnose_or_replay(record, CACHE_PATH, AGENT_LOGIC_VERSION)
        assert resolution.root_cause_code == gt.expected_root_cause_code, (
            f"{gt.order_id}: expected {gt.expected_root_cause_code}, got {resolution.root_cause_code}"
        )
        assert resolution.quantified_delta_paise == 0, (
            f"{gt.order_id}: cutoff_drift must have zero money delta, got {resolution.quantified_delta_paise}"
        )
        assert debug["hop_count"] <= MAX_TOOL_CALLS


def test_refund_clawback_resolved_correctly(settlement_queue, ground_truth_by_order_id):
    """refund_clawback isn't named in the plan's literal acceptance-criteria
    bullet, but it's an agent_resolved category with its own root cause code
    in ground_truth.json, so it's included here (flagged as an addition
    during Step 2, not silently expanded).

    Why the exact-delta assertion here proves MDR non-reversal, not just a
    label match: src/data/generator.py sets expected_delta_paise to exactly
    to_paise(refund_amount) -- the generator computes MDR/GST/TDS normally
    against the original gross amount and never touches those figures when
    it bolts on the refund, so the *correct* delta is the raw refund amount
    with no MDR/GST/TDS folded in either direction. All 3 real records here
    are non-UPI rails (credit_card, amex, netbanking) with nonzero MDR
    (verified: Rs 15.80 / 83.04 / 17.87), so if the agent wrongly reasoned
    that MDR should be subtracted or added back into the clawback, its
    number would land on refund_amount +/- mdr +/- gst +/- tds -- which is
    provably different from refund_amount for all 3 records, not a
    coincidental match. The exact-match assertion below is therefore a real
    arithmetic check on the non-reversal rule, not merely a label check.
    """
    records = _settled_by_category(settlement_queue, ground_truth_by_order_id, "refund_clawback")
    assert len(records) == 3
    for record in records:
        gt = ground_truth_by_order_id[record.order_context.order_id]
        resolution, debug, _replayed = diagnose_or_replay(record, CACHE_PATH, AGENT_LOGIC_VERSION)
        assert resolution.root_cause_code == gt.expected_root_cause_code, (
            f"{gt.order_id}: expected {gt.expected_root_cause_code}, got {resolution.root_cause_code}"
        )
        assert resolution.quantified_delta_paise == gt.expected_delta_paise, (
            f"{gt.order_id}: expected delta {gt.expected_delta_paise} (== raw refund_amount, "
            f"MDR/GST/TDS not reversed), got {resolution.quantified_delta_paise}"
        )
        assert debug["hop_count"] <= MAX_TOOL_CALLS


# ---------------------------------------------------------------------------
# LIVE tests 7-9 -- honest_exception categories, verified through the actual
# agent call (not just Layer 2's pre-agent queue routing, already covered by
# test_fast_path.py).
# ---------------------------------------------------------------------------

def test_orphan_reaches_honest_exception_via_agent(unmatched_queue, ground_truth_by_order_id):
    records = _unmatched_by_category(unmatched_queue, ground_truth_by_order_id, "orphan")
    assert len(records) == 8
    for record in records:
        resolution, debug, _replayed = diagnose_or_replay(record, CACHE_PATH, AGENT_LOGIC_VERSION)
        assert resolution.root_cause_code == "UNRESOLVED", (
            f"orphan {record.bank_credits[0].utr}: agent resolved to {resolution.root_cause_code}, expected UNRESOLVED"
        )
        assert debug["hop_count"] <= MAX_TOOL_CALLS


def test_short_settlement_reaches_honest_exception_via_agent(settlement_queue, ground_truth_by_order_id):
    records = _settled_by_category(settlement_queue, ground_truth_by_order_id, "short_settlement")
    assert len(records) == 2
    for record in records:
        assert record.bank_credits == [], "short_settlement record should carry zero bank credits by construction"
        resolution, debug, _replayed = diagnose_or_replay(record, CACHE_PATH, AGENT_LOGIC_VERSION)
        assert resolution.root_cause_code == "UNRESOLVED", (
            f"{record.order_context.order_id}: agent resolved to {resolution.root_cause_code}, expected UNRESOLVED"
        )
        assert debug["hop_count"] <= MAX_TOOL_CALLS


def test_duplicate_credit_reaches_honest_exception_via_agent(settlement_queue, ground_truth_by_order_id):
    records = _settled_by_category(settlement_queue, ground_truth_by_order_id, "duplicate_credit")
    assert len(records) == 2
    for record in records:
        assert len(record.bank_credits) == 2, (
            f"{record.order_context.order_id}: expected 2 ambiguous bank credits, got {len(record.bank_credits)}"
        )
        resolution, debug, _replayed = diagnose_or_replay(record, CACHE_PATH, AGENT_LOGIC_VERSION)
        # UNRESOLVED is itself proof no arbitrary pick was made -- a resolved
        # AgentResolution has no field that could represent "picked candidate A
        # over B" other than root_cause_code, so ruling out anything but
        # UNRESOLVED rules out force-picking either candidate.
        assert resolution.root_cause_code == "UNRESOLVED", (
            f"{record.order_context.order_id}: agent resolved to {resolution.root_cause_code} instead of "
            "declining the ambiguous duplicate credit"
        )
        assert debug["hop_count"] <= MAX_TOOL_CALLS


# ---------------------------------------------------------------------------
# DETERMINISTIC test 10 -- the 3-tool-call cap, tested as a control-flow
# property of graph.py's own code, not inferred from a live run happening to
# stay under the cap.
# ---------------------------------------------------------------------------

def test_tool_call_cap_enforced_structurally():
    """White-box test: even when the model requests a 4th tool call,
    invoke_tool must not execute it once hop_count has reached max_hops.
    No network call -- proves the cap lives in the graph's control flow,
    not in a prompt instruction the model could ignore."""

    class _StubModel:
        def invoke(self, messages):
            return AIMessage(
                content="",
                tool_calls=[{"name": "get_tax_rules", "args": {"as_of": "2025-01-01"}, "id": "call_4"}],
            )

    state = _fresh_state(hop_count=MAX_TOOL_CALLS)
    result = _invoke_tool_logic(state, _StubModel())

    assert result["status"] == "READY_TO_PROPOSE"
    assert result["hop_count"] == MAX_TOOL_CALLS
    assert result["tool_call_history"] == []


# ---------------------------------------------------------------------------
# DETERMINISTIC tests 12-13, 16 (+1 bonus) -- malformed-payload retry/
# fallback and the json-pseudo-tool safety net, all via injected stub models.
# None of these call _build_model(), bind_tools(), or the network.
# ---------------------------------------------------------------------------

def test_malformed_payload_retried_then_honest_exception():
    calls: list[list] = []

    class _StubModel:
        def invoke(self, messages):
            calls.append(messages)
            if len(calls) == 1:
                return AIMessage(content="not valid json at all")
            return AIMessage(content='{"root_cause_code": "NOT_A_REAL_CODE"}')  # invalid literal, missing fields

    state = _fresh_state()
    result = _propose_resolution_logic(state, _StubModel())

    assert result["status"] == "HONEST_EXCEPTION"
    assert len(calls) == 2, "must retry exactly once, not zero or more than once"
    retry_text = " ".join(getattr(m, "content", "") for m in calls[1])
    assert "failed schema validation" in retry_text
    assert "no JSON object found" in retry_text, "the retry message must carry attempt 1's exact parse error"


def test_malformed_then_valid_recovers():
    calls: list[int] = []
    good_json = json.dumps(
        {
            "root_cause_code": "UNRESOLVED",
            "quantified_delta_paise": 0,
            "evidence_tool_calls": [],
            "confidence_note": "recovered on retry",
        }
    )

    class _StubModel:
        def invoke(self, messages):
            calls.append(1)
            if len(calls) == 1:
                return AIMessage(content="not valid json")
            return AIMessage(content=good_json)

    state = _fresh_state()
    result = _propose_resolution_logic(state, _StubModel())

    assert result["status"] == "RESOLVED"
    assert result["resolution"].root_cause_code == "UNRESOLVED"
    assert len(calls) == 2


def test_invoke_tool_handles_json_pseudo_tool_error():
    """Regression test for the exact bug found in the adversarial_trap spike
    (see module docstring): openai/gpt-oss-120b sometimes emits a call to a
    synthetic 'json' tool not in the bound tool list, which Groq rejects
    with a 400. invoke_tool must treat this as 'no tool call, ready to
    propose' rather than crashing the record."""
    resp = httpx.Response(400, request=httpx.Request("POST", "http://x"), json={"error": {"message": "boom"}})
    err = BadRequestError(
        "Tool call validation failed: tool call validation failed: attempted to call tool 'json' "
        "which was not in request.tools",
        response=resp,
        body=None,
    )

    class _StubModel:
        def invoke(self, messages):
            raise err

    state = _fresh_state()
    result = _invoke_tool_logic(state, _StubModel())

    assert result["status"] == "READY_TO_PROPOSE"
    assert result["hop_count"] == 0
    assert result["tool_call_history"] == []


def test_invoke_tool_reraises_unrelated_api_errors():
    """The safety net above is deliberately narrow -- it must not swallow a
    400 for an unrelated reason (a real bug in tool args, a genuine bad
    request), only the specific json-pseudo-tool pattern."""
    resp = httpx.Response(400, request=httpx.Request("POST", "http://x"), json={"error": {"message": "boom"}})
    err = BadRequestError("some other 400 error unrelated to the json pseudo-tool bug", response=resp, body=None)

    class _StubModel:
        def invoke(self, messages):
            raise err

    state = _fresh_state()
    with pytest.raises(BadRequestError):
        _invoke_tool_logic(state, _StubModel())


# ---------------------------------------------------------------------------
# LIVE test 14 -- stateless execution, concrete cross-contamination
# construction (not just "runs twice without crashing").
# ---------------------------------------------------------------------------

def test_stateless_no_cross_record_leakage(settlement_queue, ground_truth_by_order_id):
    fee_drift_records = _settled_by_category(settlement_queue, ground_truth_by_order_id, "fee_drift")
    tax_records = _settled_by_category(settlement_queue, ground_truth_by_order_id, "missing_tax_line")
    record1, record2 = fee_drift_records[0], tax_records[0]
    gt1 = ground_truth_by_order_id[record1.order_context.order_id]
    gt2 = ground_truth_by_order_id[record2.order_context.order_id]
    assert gt1.expected_root_cause_code != gt2.expected_root_cause_code, (
        "precondition: record1 and record2 must have different correct answers, "
        "otherwise leakage would be undetectable"
    )

    # Both routed through the cache like every other test here. On the first
    # populate pass (empty cache) both run live in this process, genuinely
    # exercising statelessness; on a later replay-only pass this test can't
    # re-detect a regression (no live call happens) -- an accepted tradeoff
    # of banking results against Groq's daily token cap (see run_log.py).
    diagnose_or_replay(record1, CACHE_PATH, AGENT_LOGIC_VERSION)  # run/replay first; must not leak into record2
    resolution2, debug2, _replayed2 = diagnose_or_replay(record2, CACHE_PATH, AGENT_LOGIC_VERSION)

    assert resolution2.root_cause_code == gt2.expected_root_cause_code, (
        f"record2 ({gt2.order_id}) resolved to {resolution2.root_cause_code}, expected its own "
        f"{gt2.expected_root_cause_code} -- possible cross-record leakage from record1 ({gt1.order_id})"
    )
    assert resolution2.quantified_delta_paise == gt2.expected_delta_paise
    assert record1.order_context.order_id not in resolution2.confidence_note
    assert debug2["hop_count"] <= MAX_TOOL_CALLS


# ---------------------------------------------------------------------------
# DETERMINISTIC test 15 -- structural regression guard, no LLM call at all.
# ---------------------------------------------------------------------------

def test_system_prompt_excludes_final_answer_schema():
    """Regression guard for the exact bug found in the adversarial_trap
    spike (see module docstring): the final-answer JSON schema must never
    reappear in the tool-bound SYSTEM_PROMPT, only in FINAL_ANSWER_INSTRUCTION
    (shown to the model only after tools are no longer bound)."""
    for marker in ('"root_cause_code"', '"quantified_delta_paise"', '"evidence_tool_calls"'):
        assert marker not in graph.SYSTEM_PROMPT, f"{marker} leaked back into the tool-bound SYSTEM_PROMPT"
        assert marker in graph.FINAL_ANSWER_INSTRUCTION
