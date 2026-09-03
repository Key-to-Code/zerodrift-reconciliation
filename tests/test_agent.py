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
import os
import re
from pathlib import Path

import httpx
import pytest
from dotenv import load_dotenv
from groq import BadRequestError, RateLimitError
from langchain_core.messages import AIMessage

load_dotenv()

from src.agent import graph
from src.agent.discrepancy import (
    BankCredit,
    DiscrepancyRecord,
    OrderContext,
    build_settlement_discrepancy_queue,
    build_unmatched_bank_line_queue,
)
from src.agent.graph import (
    AGENT_LOGIC_VERSION,
    AgentState,
    MAX_TOOL_CALLS,
    _invoke_tool_logic,
    _invoke_with_backoff,
    _propose_resolution_logic,
    gatekeeper_check,
)
from src.agent.rate_limiter import AgentRateLimitedError
from src.agent.resolution import AgentResolution
from src.agent.run_log import (
    append_run_log,
    average_real_tokens_per_live_call,
    count_live_calls_needed,
    diagnose_or_replay,
)
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
        "tokens_used": 0,
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
# DETERMINISTIC tests -- _invoke_with_backoff's TPD-vs-TPM branching
# (src/agent/rate_limiter.py, added 2026-09-03 after a live debugging
# session found a daily-quota 429 propagating uncaught to a bare 500 at the
# API layer). No network -- a stub model raises a pre-built groq error.
# ---------------------------------------------------------------------------

def _rate_limit_error(message: str) -> RateLimitError:
    resp = httpx.Response(429, request=httpx.Request("POST", "http://x"), json={"error": {"message": message}})
    return RateLimitError(message, response=resp, body=None)


def test_backoff_raises_immediately_on_daily_quota_429_without_retrying():
    calls: list[int] = []

    class _StubModel:
        def invoke(self, messages):
            calls.append(1)
            raise _rate_limit_error(
                "Rate limit reached ... on tokens per day (TPD): Limit 200000, Used 198791, Requested 2195."
            )

    with pytest.raises(AgentRateLimitedError):
        _invoke_with_backoff(_StubModel(), [], max_retries=4)
    assert len(calls) == 1, "a daily-quota 429 must not be retried -- it will not clear in a few seconds"


def test_backoff_still_retries_per_minute_429_as_before(monkeypatch):
    monkeypatch.setattr("src.agent.graph.time.sleep", lambda seconds: None)  # skip the real 5s/10s backoff delay
    calls: list[int] = []

    class _StubModel:
        def invoke(self, messages):
            calls.append(1)
            if len(calls) < 3:
                raise _rate_limit_error(
                    "Rate limit reached ... on tokens per minute (TPM): Limit 8000, Used 7912, Requested 500."
                )
            return AIMessage(content="ok", tool_calls=[])

    result = _invoke_with_backoff(_StubModel(), [], max_retries=4)
    assert result.content == "ok"
    assert len(calls) == 3, "a per-minute 429 must still be retried with backoff, unchanged from before"


# ---------------------------------------------------------------------------
# DETERMINISTIC tests -- count_live_calls_needed / average_real_tokens_per_live_call
# (src/agent/run_log.py, added 2026-09-03 for run_batch's pre-flight budget
# check). Pure file/dict lookups against a tmp_path cache file -- no network.
# ---------------------------------------------------------------------------

def _tiny_record(order_id: str, gross_amount_paise: int = 100_000) -> DiscrepancyRecord:
    return DiscrepancyRecord(
        discrepancy_reason="fee_drift",
        order_context=OrderContext(
            order_id=order_id,
            gross_amount_paise=gross_amount_paise,
            payment_method="amex",
            timestamp="2025-01-06T10:00:00+05:30",
            refund_amount_paise=None,
        ),
    )


def _resolution(root_cause_code: str = "AMEX_SURCHARGE") -> AgentResolution:
    return AgentResolution(
        root_cause_code=root_cause_code, quantified_delta_paise=100, evidence_tool_calls=[], confidence_note="stub"
    )


def test_count_live_calls_needed_all_uncached(tmp_path):
    records = [_tiny_record("ORD_A"), _tiny_record("ORD_B")]
    empty_cache = tmp_path / "empty.jsonl"
    assert count_live_calls_needed(records, empty_cache, logic_version=1) == 2


def test_count_live_calls_needed_excludes_real_cache_hits(tmp_path):
    records = [_tiny_record("ORD_A"), _tiny_record("ORD_B")]
    cache_path = tmp_path / "cache.jsonl"
    append_run_log(cache_path, records[0], _resolution(), {"tokens_used": 500}, logic_version=1)
    assert count_live_calls_needed(records, cache_path, logic_version=1) == 1  # ORD_A cached, ORD_B is not


def test_count_live_calls_needed_stale_logic_version_counts_as_a_miss(tmp_path):
    records = [_tiny_record("ORD_A")]
    cache_path = tmp_path / "cache.jsonl"
    append_run_log(cache_path, records[0], _resolution(), {"tokens_used": 500}, logic_version=1)
    assert count_live_calls_needed(records, cache_path, logic_version=2) == 1  # cached under an old logic version


def test_count_live_calls_needed_changed_content_counts_as_a_miss(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    append_run_log(cache_path, _tiny_record("ORD_A", gross_amount_paise=100_000), _resolution(), {"tokens_used": 500}, logic_version=1)
    changed_record = _tiny_record("ORD_A", gross_amount_paise=999_999)  # same order_id, different content
    assert count_live_calls_needed([changed_record], cache_path, logic_version=1) == 1


def test_average_real_tokens_per_live_call_computes_real_average(tmp_path):
    cache_path = tmp_path / "cache.jsonl"
    append_run_log(cache_path, _tiny_record("ORD_A"), _resolution(), {"tokens_used": 6000}, logic_version=1)
    append_run_log(cache_path, _tiny_record("ORD_B"), _resolution(), {"tokens_used": 8000}, logic_version=1)
    assert average_real_tokens_per_live_call([cache_path]) == 7000.0


def test_average_real_tokens_per_live_call_ignores_zero_token_entries(tmp_path):
    """A zero tokens_used entry means genuinely never measured (e.g. a stub
    used in an offline test), not a real free call -- must not drag the
    real average toward zero."""
    cache_path = tmp_path / "cache.jsonl"
    append_run_log(cache_path, _tiny_record("ORD_A"), _resolution(), {"tokens_used": 6000}, logic_version=1)
    append_run_log(cache_path, _tiny_record("ORD_B"), _resolution(), {"tokens_used": 0}, logic_version=1)
    assert average_real_tokens_per_live_call([cache_path]) == 6000.0


def test_average_real_tokens_per_live_call_falls_back_to_default_when_no_real_data():
    empty_cache = Path("/nonexistent/does/not/exist.jsonl")
    assert average_real_tokens_per_live_call([empty_cache], default=1234.0) == 1234.0


def test_gatekeeper_rejects_evidence_not_backed_by_real_tool_history():
    """Layer 4 review finding #2, fixed at AGENT_LOGIC_VERSION 6:
    gatekeeper_check must trust state["tool_call_history"] (the graph's own
    authoritative record of what invoke_tool actually executed), never
    resolution.evidence_tool_calls (a field the model fills in itself inside
    its final JSON answer and could fabricate independently of what it
    actually called). No network call -- gatekeeper_check is pure
    state-in/state-out, no model involved."""
    fabricated_resolution = AgentResolution(
        root_cause_code="AMEX_SURCHARGE",
        quantified_delta_paise=1000,
        evidence_tool_calls=["get_tax_rules"],
        confidence_note="claims evidence it never actually gathered",
    )

    # Case 1: model claims evidence_tool_calls=["get_tax_rules"] but the
    # graph's real tool_call_history is entirely empty -- no tool was ever
    # called for this record at all.
    empty_history_state = _fresh_state(
        status="PROCESSING", resolution=fabricated_resolution, tool_call_history=[]
    )
    result = gatekeeper_check(empty_history_state)
    assert result["status"] == "HONEST_EXCEPTION", (
        "a non-UNRESOLVED resolution with zero real tool calls must not be postable, "
        "regardless of what evidence_tool_calls claims"
    )
    assert "gatekeeper_rejected" in result["raw_failure"]

    # Case 2: real tool_call_history is non-empty (a different tool was
    # genuinely called) but doesn't contain the specific tool claimed as
    # evidence -- the claimed evidence is still fabricated even though some
    # real tool activity happened.
    wrong_tool_state = _fresh_state(
        status="PROCESSING",
        resolution=fabricated_resolution,
        tool_call_history=[{"name": "check_settlement_timing", "args": {}, "result": {}}],
    )
    result2 = gatekeeper_check(wrong_tool_state)
    assert result2["status"] == "HONEST_EXCEPTION", (
        "claimed evidence_tool_calls must be a subset of the tools actually recorded "
        "in state['tool_call_history'], not merely 'some tool was called'"
    )
    assert "gatekeeper_rejected" in result2["raw_failure"]

    # Control: real tool_call_history genuinely contains the claimed tool ->
    # must be accepted, proving the fix isn't just rejecting everything.
    real_history_state = _fresh_state(
        status="PROCESSING",
        resolution=fabricated_resolution,
        tool_call_history=[{"name": "get_tax_rules", "args": {}, "result": {}}],
    )
    result3 = gatekeeper_check(real_history_state)
    assert result3["status"] == "RESOLVED"


def test_tokens_used_summed_from_real_usage_metadata():
    """Budgeting fix: debug_info['tokens_used'] (surfaced via
    diagnose_discrepancy) must be the real Groq-reported usage.total_tokens
    summed across every model call made for a record, not an estimate.
    Exercises both _invoke_tool_logic (one tool hop) and
    _propose_resolution_logic (a malformed-then-valid retry) accumulating
    into the same running total via stub AIMessages carrying
    usage_metadata -- no network call, so the number asserted here isn't
    itself a live cost."""

    class _ToolStubModel:
        def invoke(self, messages):
            return AIMessage(
                content="",
                tool_calls=[{"name": "get_tax_rules", "args": {"as_of": "2025-01-01"}, "id": "call_1"}],
                usage_metadata={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150},
            )

    state_after_tool = _invoke_tool_logic(_fresh_state(), _ToolStubModel())
    assert state_after_tool["tokens_used"] == 150

    propose_calls: list[int] = []

    class _ProposeStubModel:
        def invoke(self, messages):
            propose_calls.append(1)
            if len(propose_calls) == 1:
                return AIMessage(
                    content="not valid json",
                    usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
                )
            good_json = json.dumps(
                {
                    "root_cause_code": "UNRESOLVED",
                    "quantified_delta_paise": 0,
                    "evidence_tool_calls": [],
                    "confidence_note": "ok",
                }
            )
            return AIMessage(
                content=good_json,
                usage_metadata={"input_tokens": 20, "output_tokens": 10, "total_tokens": 30},
            )

    # Chained onto state_after_tool -- proves accumulation carries across
    # nodes within one record's diagnostic run, not just within one call.
    final_state = _propose_resolution_logic(state_after_tool, _ProposeStubModel())
    assert final_state["status"] == "RESOLVED"
    assert final_state["tokens_used"] == 150 + 15 + 30, (
        "must equal the tool-hop cost plus BOTH propose_resolution attempts "
        "(the failed first try's tokens were still real Groq spend, not free)"
    )


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


# ---------------------------------------------------------------------------
# DETERMINISTIC tests 16-17 -- graph.py loads .env itself now, not just this
# test file (real bug: a real GROQ_API_KEY sitting in .env never reached a
# plain `uvicorn` process, which has no test file's module-level load_dotenv()
# to do it for it -- see graph.py's own comment on the load_dotenv() call).
# Uses a throwaway temp .env, never the real committed one -- must pass on a
# machine with no real Groq key at all, same as CI.
# ---------------------------------------------------------------------------

def test_graph_module_calls_load_dotenv_and_build_model_then_succeeds(monkeypatch):
    """graph.py's own module-level load_dotenv() call is the actual fix for
    the real bug (a real GROQ_API_KEY sitting in .env never reached a plain
    `uvicorn` process, which has no test file's module-level load_dotenv()
    to do it for it). Spies on dotenv.load_dotenv itself rather than trying
    to redirect python-dotenv's own file-discovery -- confirmed by hand
    while writing this test that python-dotenv's default find_dotenv()
    resolves relative to the CALLING file's location via stack inspection
    (i.e. always finds this repo's real D:\\...\\.env, since it walks up from
    src/agent/graph.py itself), NOT os.getcwd() -- so monkeypatch.chdir()
    alone cannot redirect it, and trying to would either miss the point or
    require writing a file into the real source tree. Spying on whether
    dotenv.load_dotenv() gets called, and that a key present in os.environ
    afterward lets _build_model() succeed, tests the actual integration
    without re-testing python-dotenv's own (well-tested) file lookup."""
    import importlib

    import dotenv

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    calls = []

    def spy_load_dotenv(*args, **kwargs):
        calls.append((args, kwargs))
        monkeypatch.setenv("GROQ_API_KEY", "test-key-from-dotenv")

    monkeypatch.setattr(dotenv, "load_dotenv", spy_load_dotenv)
    try:
        importlib.reload(graph)  # re-executes `from dotenv import load_dotenv; load_dotenv()`
        assert calls, "graph.py's module-level code must call dotenv.load_dotenv()"
        assert os.environ.get("GROQ_API_KEY") == "test-key-from-dotenv"
        graph._build_model()  # constructs a ChatGroq client; must not raise -- no network call happens here
    finally:
        monkeypatch.setattr(dotenv, "load_dotenv", dotenv.load_dotenv)  # harmless no-op if never patched
        importlib.reload(graph)  # restore graph's real dotenv binding + real .env state for later tests


def test_build_model_still_raises_a_clear_error_with_no_key_anywhere(monkeypatch):
    """Regression guard for the failure mode itself: CLAUDE.md forbids
    silently faking a model client -- a genuinely missing key must still
    fail loudly, not construct a client that would error confusingly deep
    inside a live call instead. _build_model() reads os.environ directly and
    never calls load_dotenv() itself (only this module's import-time call
    does, tested above), so deleting the env var alone is a complete,
    reload-free test of this path."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY not set"):
        graph._build_model()
