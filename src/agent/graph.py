"""LangGraph bounded diagnostic loop (docs/plan.md Sec.4.3).

Model client note (binding, not a placeholder): Claude is the intended
production model for this project -- CLAUDE.md's only permitted
Razorpay-internal claim is that Agent Studio runs on Anthropic's Claude
Agent SDK, and this project is built for that track. Development and
testing in this repo, however, run against Groq-hosted openai/gpt-oss-120b
(langchain-groq) instead of the Anthropic API, for two stated reasons:
(1) cost -- Anthropic Console credits are not available for the iteration
volume this layer needs (many records x multiple retries x repeated test
runs while building), and Groq's free tier covers that volume; (2) model
choice within Groq -- Llama 3.3 70B, the originally intended dev model, was
checked against this Groq API key's live /openai/v1/models listing at
build time and is not present (Groq appears to have deprecated/rotated it
out of the free-tier offering since it was documented); openai/gpt-oss-120b
is the largest open-weights chat model this key actually has access to, so
it was used instead. Swapping back to Claude for production is a one-line
change to _build_model() below; nothing else in this module is
Groq-specific.

Nodes: classify_discrepancy -> invoke_tool (<=3 calls) -> propose_resolution
-> gatekeeper_check -> resolved | honest_exception.

Stateless per-record execution: diagnose_discrepancy() builds a fresh state
dict for every call. No module-level mutable state is read by any node.

Testability seam: invoke_tool/propose_resolution are thin wrappers that
build a real model via _build_model() and pacing, then delegate to
_invoke_tool_logic/_propose_resolution_logic, which take an already-built
model object and contain all the actual control flow. Tests call the
_logic functions directly with an injected stub model -- they never call
_build_model(), bind_tools(), or touch the network, so a test that's meant
to be a deterministic control-flow check can't silently degrade into a
live API call.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, TypedDict

from dotenv import load_dotenv
from groq import APIStatusError
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from pydantic import ValidationError

from src.agent.discrepancy import DiscrepancyRecord
from src.agent.rate_limiter import AgentRateLimitedError, daily_token_tracker, is_daily_quota_error
from src.agent.resolution import AgentResolution
from src.agent.tools import TOOL_REGISTRY, get_tax_rules, query_merchant_contract, check_settlement_timing

# python-dotenv is a declared dependency (requirements.txt) that was never
# actually wired up here -- tests/test_agent.py has always called
# load_dotenv() itself before importing this module, which is exactly why
# live tests worked while a real `uvicorn` process (no test file's
# module-level code to do it for them) hit "GROQ_API_KEY not set" below even
# with a real key sitting in .env. Belongs here, not just in main.py's
# startup, so any entrypoint that imports this module (a future evaluate.py
# script, a REPL, tests that don't already load it) gets it too. Does not
# override an already-set real environment variable (python-dotenv's
# default), so a production deployment's real env var still wins over a
# stray local .env file.
load_dotenv()

MAX_TOOL_CALLS = 3
GROQ_MODEL = "openai/gpt-oss-120b"

# Bump this by hand whenever a change to SYSTEM_PROMPT, FINAL_ANSWER_INSTRUCTION,
# a tool's behavior, or GROQ_MODEL could change what the agent produces for a
# given record. src/agent/run_log.py's cache treats a cached entry as stale
# (and re-calls the model) whenever this doesn't match the entry's logged
# version -- see that module's docstring for why (Groq's free-tier daily
# token cap makes blind re-running of the full live suite unsustainable).
#
# BEFORE spending any live budget on a bump: if the change is to pure
# post-processing logic that never touches SYSTEM_PROMPT,
# FINAL_ANSWER_INSTRUCTION, a tool's behavior, or the model (e.g. a
# gatekeeper_check rule, JSON parsing, retry/backoff logic), first replay
# every cached entry's debug_info["tool_call_history"] through the new logic
# offline (zero API calls) and check whether any status/resolution would
# change. Only fall through to a live re-run if that replay can't settle the
# question -- e.g. the fix depends on something the old cache never recorded,
# or the change does touch model-facing behavior. The v6 gatekeeper fix
# below is the worked example: an offline replay against all 37 cached
# entries found 0 flips, and that alone was accepted as sufficient
# confirmation the fix was safe (one live canary run followed only to sanity
# check the pipeline end-to-end, not to re-prove correctness the offline
# check had already shown).
#
# v2: get_tax_rules/check_settlement_timing now compute expected tax paise /
# actual business-day gap directly (see tools.py), instead of leaving that
# arithmetic to the model -- fixes 2 of 3 correctness failures found in v1's
# live test run (a TDS delta arithmetic error, a cutoff misclassification).
#
# v3 (this bump): v2's contract-clause fix for the third failure
# (AMEX_SURCHARGE vs INTL_MARKUP on ORD1069) did NOT work when re-run live --
# the model never called query_merchant_contract at all, so the clause was
# never seen. Root cause, confirmed from the cached debug_info's
# tool_call_history (zero-cost, no live call needed): get_tax_rules'
# expected_mdr_paise is the DOMESTIC standard rate regardless of
# is_international (src/data/generator.py's _compute_settlement_fields takes
# no is_international parameter -- this mirrors the generator's own "clean
# flow" baseline, which is correct; the ambiguity is real, not a bug in the
# tool's arithmetic). The model concluded AMEX_SURCHARGE purely because
# actual MDR exceeded that domestic figure, without ever checking
# settlement_context.is_international -- a field already present in the
# record it was given, not requiring any tool call. Fixed by adding an
# explicit precedence rule to SYSTEM_PROMPT instructing the model to decide
# AMEX_SURCHARGE vs INTL_MARKUP from settlement_context.is_international
# directly, never from the MDR magnitude alone. NOT YET verified live --
# Groq's daily token cap was hit again immediately after this diagnosis
# (offline, from the cache) was made; verification is pending quota.
#
# v4 (this bump): v3 re-run (fresh API key, no quota errors this time) showed
# the INTL_MARKUP/AMEX_SURCHARGE precedence fix from v3 actually worked (both
# ORD1069 and ORD1080 got the right root_cause_code live), but surfaced two
# further real bugs, both diagnosed for free from the cached debug_info:
# (a) ORD1069's delta was 5268 vs expected 4464 -- the model summed the MDR
# delta (4464, correct) AND the resulting GST-on-MDR delta (804) into one
# figure, when GST-on-MDR just moves with MDR and isn't an independent
# deviation; (b) ORD1080's delta was -346 vs expected 346 -- the model
# reported actual-minus-expected (0-346) instead of a magnitude. Fixed by
# adding explicit quantified_delta_paise semantics to SYSTEM_PROMPT: always a
# non-negative magnitude, and MDR-only (not MDR+GST) for fee/rate deviations.
# A third finding (ORD1067, a cutoff_drift record, returned UNRESOLVED with
# zero tool calls) is NOT yet understood or fixed -- flagged for a live
# retry to check whether it's a real prompt gap or ordinary agent
# non-determinism (docs/plan.md Sec.5) before touching anything for it.
#
# v5 (this bump): v4's live retry did NOT reproduce the zero-tool-call
# UNRESOLVED from before (so that particular instance looks like ordinary
# non-determinism, not a systematic gap) -- but it surfaced a different, real
# bug on the same record (ORD1067): the model got expected_window_business_days
# (1, UPI) and actual_gap_business_days (2) both correctly from
# check_settlement_timing, and reasoned about them correctly in its own
# confidence_note, but then labeled it CUTOFF_T2 -- reading the code as
# keyed to the actual observed gap (2 days) rather than the rail's own
# standard window (1 day, UPI). CUTOFF_T1/CUTOFF_T2 were never defined
# anywhere in the prompt, so this was a reasonable guess at undefined
# semantics, not an arithmetic error. Fixed by adding an explicit definition
# to SYSTEM_PROMPT: the code is keyed to expected_window_business_days, never
# to actual_gap_business_days.
#
# v6 (this bump): code-review finding, not a live-run failure -- gatekeeper_check
# was trusting resolution.evidence_tool_calls (a field the model fills in itself
# inside its own final JSON answer) to decide whether a non-UNRESOLVED resolution
# is backed by real evidence, instead of state["tool_call_history"] (the graph's
# own authoritative record of what invoke_tool actually executed). A model could
# claim evidence_tool_calls=["get_tax_rules"] without ever having called it, and
# the old gatekeeper would post it as RESOLVED anyway. Fixed by checking
# state["tool_call_history"] directly, and rejecting if the claimed tool names
# aren't a subset of what was actually called -- see
# test_gatekeeper_rejects_evidence_not_backed_by_real_tool_history.
AGENT_LOGIC_VERSION = 6

# Proactive pacing ahead of every REAL model call (not applied to the
# deterministic _logic functions below, which tests call directly without a
# real model). Groq's free tier caps at 8000 TPM on this key -- reactive
# backoff (_invoke_with_backoff) already handles a 429 gracefully, but
# spacing calls out reduces how often the cap gets hit at all during a
# dense test run touching many records back-to-back.
MODEL_CALL_PACING_SECONDS = 2.0

SYSTEM_PROMPT = """You are a financial reconciliation diagnostic agent for a payment gateway merchant's books.

You are given ONE discrepancy record. It may include:
- order_context: the internal order (gross amount, payment rail, timestamp, any refund).
- settlement_context: the gateway's settlement figures (MDR, GST on MDR, TDS under Sec 194-O,
  net amount, settlement date, whether international).
- bank_credits: 0, 1, or 2+ bank statement lines associated with this discrepancy.
- candidate_orders: orders whose settled net amount and settlement date are merely CLOSE to
  an unmatched bank credit -- proximity in amount or date is NEVER evidence of a real link.

Your job is to diagnose why this record didn't reconcile automatically:
- If order_context/settlement_context are present, call get_tax_rules WITH gross_amount_paise
  and rail filled in -- it will compute the STANDARD expected mdr/gst/tds for this exact record,
  which you then compare yourself against settlement_context's actual figures to find which line
  deviates and by how much (do not compute these percentages yourself by hand). Similarly, call
  check_settlement_timing WITH settlement_date filled in -- it will compute the actual business-day
  gap for this exact record, which you then compare yourself against expected_window_business_days
  (do not count business days by hand). Also check whether a refund_amount_paise on the order was
  properly reflected. A genuine deviation gets a specific root_cause_code and an exact
  quantified_delta_paise, decided by you from the comparison -- the tools never tell you the
  root cause directly, only the computed facts to compare.
- quantified_delta_paise is ALWAYS a non-negative magnitude: how many paise the deviation is
  worth, never a signed actual-minus-expected or expected-minus-actual value. A missing tax
  line (e.g. actual TDS 0 vs expected TDS 346) has delta 346, not -346.
- CUTOFF_T1 vs CUTOFF_T2 is named after the RAIL'S OWN STANDARD expected_window_business_days
  from check_settlement_timing, never the actual_gap_business_days you observed. CUTOFF_T1 means
  this rail's standard window is T+1 (this is only ever UPI) and it was violated -- even though
  the actual gap that violates it will be 2 or more business days, not 1. CUTOFF_T2 means this
  rail's standard window is T+2 (any non-UPI domestic rail) and it was violated. Decide which
  applies by checking expected_window_business_days, not by matching the actual gap's number.
- For a fee/rate deviation (AMEX_SURCHARGE, INTL_MARKUP), quantified_delta_paise is the MDR
  delta ONLY (actual mdr_paise minus expected_mdr_paise) -- never add the resulting GST-on-MDR
  delta on top. GST-on-MDR is 18% of whatever MDR actually is; it moves with MDR automatically
  and is not a second, independent deviation to add in.
- An MDR overage on the amex rail is ambiguous between two root causes: AMEX_SURCHARGE
  (amex-specific, domestic) and INTL_MARKUP (international, any rail). Do NOT decide this from
  the MDR numbers alone -- get_tax_rules' expected_mdr_paise is the DOMESTIC standard rate for
  the rail regardless of internationality, so an international transaction will always show an
  "overage" against it even when nothing is wrong beyond the markup itself. The deciding fact is
  settlement_context.is_international, which is already given to you in the record -- no tool
  call needed. If is_international is true, the root cause is INTL_MARKUP, never AMEX_SURCHARGE,
  regardless of rail. Only use AMEX_SURCHARGE when is_international is false.
- If there are 2+ bank_credits for the same settlement, that is an unresolved ambiguity -- you
  cannot know which credit is the real one without more information than you have. Do not guess.
- If there are 0 bank_credits for a real settlement, the money never arrived -- this cannot be
  resolved by classifying a root cause; return UNRESOLVED.
- If candidate_orders are the only thing present (no order_context/settlement_context), a
  candidate is only a genuine match if you find concrete, checkable evidence tying it to this
  specific bank credit (e.g. a UTR appearing in the narration). If no such evidence exists, you
  MUST return root_cause_code="UNRESOLVED" and quantified_delta_paise=0 -- do not force-match on
  coincidence.

You may call up to 3 read-only tools (query_merchant_contract, get_tax_rules,
check_settlement_timing) to gather evidence. None of them can confirm an order<->bank link by
themselves; they only explain fee/tax/timing deltas for a link you already have concrete
evidence for. Call a tool only if you genuinely need more evidence; if you already have enough
to decide, respond with plain text saying so and call no tool.
"""

# Deliberately kept OUT of SYSTEM_PROMPT above and only shown once tools are no
# longer bound (propose_resolution runs an untooled model call). Describing this
# schema in the same turn that also has tools bound caused openai/gpt-oss-120b on
# Groq to sometimes wrap its final JSON answer as a call to a synthetic "json"
# tool that isn't in the bound tool list, which Groq's API then rejects with a
# 400 -- a reproducible integration bug, not a model reasoning failure, found and
# fixed during the adversarial_trap spike (test_system_prompt_excludes_final_answer_schema
# in tests/test_agent.py is the permanent regression guard for it).
FINAL_ANSWER_INSTRUCTION = """Respond now with ONLY a JSON object matching this schema, no other text, no tool calls:
{"root_cause_code": "<one of AMEX_SURCHARGE|INTL_MARKUP|MISSING_GST|MISSING_TDS|CUTOFF_T1|CUTOFF_T2|BATCH_LEVEL_FEE|REFUND_NO_MDR_REVERSAL|UNRESOLVED>",
 "quantified_delta_paise": <integer>,
 "evidence_tool_calls": [<tool names you called>],
 "confidence_note": "<one sentence>"}
Remember: root_cause_code="UNRESOLVED" and quantified_delta_paise=0 unless you found
concrete evidence for a specific root cause -- proximity in amount or date alone, or an
ambiguous set of multiple bank credits, is never sufficient.
"""


class AgentState(TypedDict):
    record: DiscrepancyRecord
    hop_count: int
    max_hops: int
    tool_call_history: list[dict]
    messages: list[Any]
    resolution: AgentResolution | None
    status: str
    raw_failure: str | None
    tokens_used: int


def _build_model():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY not set. Add it to a local .env file (gitignored) -- see "
            "src/agent/graph.py module docstring for why Groq is used in dev."
        )
    return ChatGroq(model=GROQ_MODEL, api_key=api_key, temperature=0)


def _invoke_with_backoff(model, messages, max_retries: int = 4):
    """Groq's free tier enforces a per-minute token budget (observed: 8000
    TPM on this key). A 429 there is routine, not exceptional, so retry with
    backoff rather than surfacing it as a record failure. No pacing sleep in
    here -- that lives in the invoke_tool/propose_resolution node wrappers
    so the deterministic _logic functions (called directly by tests with a
    stub model) never sleep or depend on real timing.

    A DAILY-quota (TPD) 429 is a different case, NOT retried here: Groq's
    daily window doesn't clear in the few seconds this loop sleeps, so
    retrying just delays an identical failure. Raised immediately as
    AgentRateLimitedError instead (src/agent/rate_limiter.py), which
    src/api/main.py catches and turns into a clear response rather than
    letting the raw groq.RateLimitError propagate to a bare 500.
    """
    for attempt in range(max_retries):
        try:
            return model.invoke(messages)
        except APIStatusError as exc:
            if exc.status_code == 429 and is_daily_quota_error(exc):
                raise AgentRateLimitedError(str(exc)) from exc
            if exc.status_code == 429 and attempt < max_retries - 1:
                time.sleep(5 * (attempt + 1))
                continue
            raise


def _response_tokens(response) -> int:
    """Real token cost of one model call, from Groq's own usage accounting
    (langchain_groq populates AIMessage.usage_metadata from the API
    response) -- not an estimate. Returns 0 for a stub/test AIMessage built
    without usage_metadata (the deterministic _logic tests never touch the
    network, so there's genuinely nothing to report there), and for any
    response type that doesn't carry it."""
    usage = getattr(response, "usage_metadata", None)
    return usage.get("total_tokens", 0) if usage else 0


def _record_to_prompt(record: DiscrepancyRecord) -> str:
    payload = {
        "discrepancy_reason": record.discrepancy_reason,
        "order_context": record.order_context.model_dump() if record.order_context else None,
        "settlement_context": record.settlement_context.model_dump() if record.settlement_context else None,
        "bank_credits": [b.model_dump() for b in record.bank_credits],
        "candidate_orders": [c.model_dump() for c in record.candidate_orders],
        "batch_context": record.batch_context.model_dump() if record.batch_context else None,
    }
    return f"Discrepancy record:\n{json.dumps(payload, indent=2)}"


def classify_discrepancy(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=_record_to_prompt(state["record"]))]
    return {**state, "messages": messages, "status": "PROCESSING"}


def _invoke_tool_logic(state: AgentState, model) -> AgentState:
    try:
        response = _invoke_with_backoff(model, state["messages"])
    except APIStatusError as exc:
        if "not in request.tools" in str(exc):
            # openai/gpt-oss-120b occasionally wraps a premature final answer
            # as a call to a nonexistent "json" tool instead of just replying
            # with plain text. Treat it the same as "no tool call requested" --
            # drop the malformed turn and let propose_resolution ask again
            # explicitly, with tools no longer bound.
            return {**state, "status": "READY_TO_PROPOSE"}
        raise

    call_tokens = _response_tokens(response)
    daily_token_tracker.record_usage(call_tokens)
    tokens_used = state.get("tokens_used", 0) + call_tokens
    messages = state["messages"] + [response]

    if not response.tool_calls or state["hop_count"] >= state["max_hops"]:
        return {**state, "messages": messages, "status": "READY_TO_PROPOSE", "tokens_used": tokens_used}

    hop_count = state["hop_count"]
    tool_call_history = list(state["tool_call_history"])
    for call in response.tool_calls:
        if hop_count >= state["max_hops"]:
            break
        fn = TOOL_REGISTRY[call["name"]]
        result = fn(**call["args"])
        tool_call_history.append({"name": call["name"], "args": call["args"], "result": result})
        messages.append(ToolMessage(content=json.dumps(result), tool_call_id=call["id"]))
        hop_count += 1

    return {
        **state,
        "messages": messages,
        "hop_count": hop_count,
        "tool_call_history": tool_call_history,
        "status": "PROCESSING",
        "tokens_used": tokens_used,
    }


def invoke_tool(state: AgentState) -> AgentState:
    time.sleep(MODEL_CALL_PACING_SECONDS)
    model = _build_model().bind_tools([query_merchant_contract, get_tax_rules, check_settlement_timing])
    return _invoke_tool_logic(state, model)


def _route_after_tool(state: AgentState) -> str:
    if state["status"] == "READY_TO_PROPOSE":
        return "propose_resolution"
    return "invoke_tool"


def _propose_resolution_logic(state: AgentState, model) -> AgentState:
    messages = state["messages"] + [HumanMessage(content=FINAL_ANSWER_INSTRUCTION)]
    tokens_used = state.get("tokens_used", 0)

    for attempt in range(2):
        response = _invoke_with_backoff(model, messages)
        daily_token_tracker.record_usage(_response_tokens(response))
        tokens_used += _response_tokens(response)
        raw = response.content if isinstance(response.content, str) else str(response.content)
        try:
            resolution = AgentResolution.model_validate_json(_extract_json(raw))
            return {
                **state,
                "resolution": resolution,
                "status": "RESOLVED",
                "messages": messages + [response],
                "tokens_used": tokens_used,
            }
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            if attempt == 0:
                messages = messages + [
                    response,
                    HumanMessage(
                        content=(
                            f"Your last response failed schema validation with this exact error:\n{exc}\n"
                            "Respond again with ONLY a corrected JSON object matching the schema."
                        )
                    ),
                ]
                continue
            return {
                **state,
                "status": "HONEST_EXCEPTION",
                "raw_failure": f"malformed_output_after_retry: {exc}\nraw_response: {raw!r}",
                "tokens_used": tokens_used,
            }
    return {**state, "status": "HONEST_EXCEPTION", "raw_failure": "unreachable", "tokens_used": tokens_used}


def propose_resolution(state: AgentState) -> AgentState:
    time.sleep(MODEL_CALL_PACING_SECONDS)
    model = _build_model()
    return _propose_resolution_logic(state, model)


def _extract_json(text: str) -> str:
    text = text.strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object found in model output: {text!r}")
    return text[start : end + 1]


def gatekeeper_check(state: AgentState) -> AgentState:
    if state["status"] == "HONEST_EXCEPTION":
        return state
    resolution = state["resolution"]
    # A resolution that claims a real root cause must be backed by tools the
    # graph itself actually invoked. state["tool_call_history"] is the
    # authoritative record -- built by _invoke_tool_logic from real tool
    # executions -- never resolution.evidence_tool_calls, which is just the
    # model's own self-report inside its final JSON answer and could claim a
    # tool it never actually called (Layer 4 review finding #2). (Design
    # decision, not in the plan's literal text; flagged here rather than
    # added silently.)
    actually_called = {call["name"] for call in state["tool_call_history"]}
    claimed_but_not_called = set(resolution.evidence_tool_calls) - actually_called
    if resolution.root_cause_code != "UNRESOLVED" and (not actually_called or claimed_but_not_called):
        return {
            **state,
            "status": "HONEST_EXCEPTION",
            "raw_failure": (
                "gatekeeper_rejected: non-UNRESOLVED resolution not backed by state['tool_call_history'] "
                f"(actually_called={sorted(actually_called)}, claimed={resolution.evidence_tool_calls})"
            ),
        }
    final_status = "HONEST_EXCEPTION" if resolution.root_cause_code == "UNRESOLVED" else "RESOLVED"
    return {**state, "status": final_status}


def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("classify_discrepancy", classify_discrepancy)
    graph.add_node("invoke_tool", invoke_tool)
    graph.add_node("propose_resolution", propose_resolution)
    graph.add_node("gatekeeper_check", gatekeeper_check)

    graph.set_entry_point("classify_discrepancy")
    graph.add_edge("classify_discrepancy", "invoke_tool")
    graph.add_conditional_edges(
        "invoke_tool", _route_after_tool, {"invoke_tool": "invoke_tool", "propose_resolution": "propose_resolution"}
    )
    graph.add_edge("propose_resolution", "gatekeeper_check")
    graph.add_edge("gatekeeper_check", END)
    return graph.compile()


_diagnostic_workflow = None


def diagnose_discrepancy(record: DiscrepancyRecord) -> tuple[AgentResolution, dict]:
    """Runs the bounded diagnostic loop for exactly one record, in total
    isolation from any other record (fresh state dict every call -- no
    cross-record memory). Returns (resolution, debug_info) where debug_info
    carries hop_count/tool_call_history/status/raw_failure/tokens_used for
    logging and test assertions.

    Raises AgentRateLimitedError (src/agent/rate_limiter.py), with zero
    network calls, if this process's own tracked usage today is already
    within the safety margin of Groq's daily token cap -- callers (Layer 6's
    trigger_batch_run) catch this and return a clear, honest response
    instead of an opaque 500. Only reached on a live call: diagnose_or_replay
    (src/agent/run_log.py) checks the cache first and never calls this
    function at all on a cache hit, so the frozen dataset's fully-cached
    path is unaffected by quota regardless of remaining budget.

    tokens_used is real usage.total_tokens summed across every live model
    call made for this record (classify+invoke_tool hops and both
    propose_resolution attempts), read from Groq's own response accounting
    -- never estimated. This is what makes a real "how much have we actually
    spent" answer possible by summing data/agent_runs/*.jsonl, instead of
    the ~4,400/~11,700-per-record-run *estimates* the project ran on before
    this was added (see AGENT_LOGIC_VERSION's docstring history above).
    """
    daily_token_tracker.check_budget()

    global _diagnostic_workflow
    if _diagnostic_workflow is None:
        _diagnostic_workflow = _build_graph()

    initial_state: AgentState = {
        "record": record,
        "hop_count": 0,
        "max_hops": MAX_TOOL_CALLS,
        "tool_call_history": [],
        "messages": [],
        "resolution": None,
        "status": "PROCESSING",
        "raw_failure": None,
        "tokens_used": 0,
    }
    final_state = _diagnostic_workflow.invoke(initial_state)

    if final_state["status"] == "HONEST_EXCEPTION":
        resolution = AgentResolution(
            root_cause_code="UNRESOLVED",
            quantified_delta_paise=0,
            evidence_tool_calls=[h["name"] for h in final_state["tool_call_history"]],
            confidence_note=final_state.get("raw_failure") or "routed to honest_exception by gatekeeper",
        )
    else:
        resolution = final_state["resolution"]

    debug_info = {
        "hop_count": final_state["hop_count"],
        "tool_call_history": final_state["tool_call_history"],
        "status": final_state["status"],
        "raw_failure": final_state.get("raw_failure"),
        "tokens_used": final_state.get("tokens_used", 0),
    }
    return resolution, debug_info
