"""Read-only tools available to the Layer 4 agent (docs/plan.md Sec.4.2).

No tool writes to the ledger and no tool marks anything resolved -- the
agent proposes an AgentResolution, gatekeeper_check decides whether it's
postable. Every tool call is logged to state["tool_call_history"] by
graph.py's invoke_tool node, which is what enforces the 3-call hard cap
structurally (the cap lives in the graph, not in a prompt instruction).

query_merchant_contract's schedule is SYNTHETIC -- there is no merchant_id
field anywhere in the frozen dataset (src/data/generator.py never emits
one), so this models a single default merchant's contract terms rather than
looking one up. Labelled here per CLAUDE.md Sec.1 ("never invent sample
data that isn't explicitly labelled synthetic").

get_tax_rules() and check_settlement_timing() were extended after a live
test run (tests/test_agent.py) found 3 real correctness failures, all
tracing to one design mistake: the original tools returned rates and rules
only, and expected the model to apply paise-precision arithmetic
(get_tax_rules) or IST business-day calendar arithmetic across weekends and
holidays (check_settlement_timing) on top of them itself -- exactly the
kind of thing an LLM is unreliable at. The fix moves that arithmetic into
the tools, reusing already-tested Layer 1/2 code
(src.matching.fast_path.expected_mdr_gst_tds_paise,
src.common.calendar.business_days_between) rather than writing new
arithmetic here.

Deliberate line, not crossed: these tools return computed FACTS (expected
standard mdr/gst/tds for this gross amount and rail; the actual business-day
gap between an order and a settlement date), never a root_cause_code or any
other diagnostic conclusion. The model still has to pull the settlement's
ACTUAL mdr/gst/tds/net figures out of the discrepancy record it already has,
diff them against what these tools report as expected, decide which specific
line deviates and what that means, and choose the root_cause_code itself.
Handing back the diagnosis directly (e.g. an "is_late"/"violation_type"
field, or a tool that takes the actual figures and returns the delta) would
turn the agent into a lookup wrapper around deterministic code -- that
line is deliberately not crossed here.
"""
from __future__ import annotations

from datetime import date, datetime

from src.common.calendar import business_days_between
from src.data.generator import _settlement_window_business_days
from src.matching.fast_path import expected_mdr_gst_tds_paise

TDS_RATE_EFFECTIVE_FROM = date(2024, 10, 1)

# SYNTHETIC contract data -- not derived from the frozen dataset, which has
# no merchant_id. Rates mirror src/data/generator.py's STANDARD_MDR_RATES so
# a "clean" record and a fee_drift record are distinguishable by comparing
# actuals against this schedule.
_SYNTHETIC_CONTRACT_SCHEDULE = {
    "merchant_id": "m_synthetic_0001",
    "effective_date": "2025-01-01",
    "pricing_model": "TIERED_BLENDED",
    "schedules": [
        {"rail": "credit_card", "base_rate": "0.018", "international_markup": "0.010", "minimum_fee_paise": 0},
        {"rail": "debit_card", "base_rate": "0.009", "international_markup": "0.010", "minimum_fee_paise": 0},
        {"rail": "amex", "base_rate": "0.025", "international_markup": "0.010", "minimum_fee_paise": 500},
        {"rail": "netbanking", "base_rate": "0.012", "international_markup": "0.0", "minimum_fee_paise": 0},
        {"rail": "upi", "base_rate": "0.000", "p2m_interchange_cap_paise": 0},
    ],
    "clauses": (
        "Amex transactions carry an additional surcharge over the standard rate "
        "per issuer agreement. International transactions on any card rail incur "
        "an additional markup over the domestic base rate. When a transaction is "
        "BOTH on the Amex rail AND international, classify it as an international "
        "markup, not an Amex surcharge -- internationality takes precedence over "
        "rail-specific surcharge classification. UPI P2M MDR is nil by regulation "
        "(RBI circular) -- mdr and gst_on_mdr are always zero on UPI."
    ),
}


def query_merchant_contract(payment_method: str) -> dict:
    return _SYNTHETIC_CONTRACT_SCHEDULE


def get_tax_rules(as_of: str, gross_amount_paise: int | None = None, rail: str | None = None) -> dict:
    """as_of: ISO date string. TDS under Sec 194-O is 0.1%, effective
    2024-10-01 (reduced from 1%) -- CLAUDE.md Sec.4. Only this rate is
    modeled; the frozen dataset's window (Jan 2025) is entirely after the
    change, so no pre-change branch is exercised, but the tool still
    reports effective_from so a test can assert the correct rate is picked.

    gross_amount_paise/rail (optional): when both are given, also returns
    the STANDARD expected MDR/GST/TDS in paise for that specific amount and
    rail, computed via the same rate table via
    src.matching.fast_path.expected_mdr_gst_tds_paise (already tested in
    Layer 2). This is a computed fact, not a diagnosis -- compare it
    yourself against the settlement's actual mdr_paise/gst_on_mdr_paise/
    tds_paise (already in the discrepancy record) to find which line
    deviates and by how much.
    """
    result = {
        "gst_rate": "0.18",
        "tds_rate": "0.001",
        "effective_from": TDS_RATE_EFFECTIVE_FROM.isoformat(),
        "notes": "TDS under Sec 194-O reduced from 1% to 0.1% effective 2024-10-01.",
    }
    if gross_amount_paise is not None and rail is not None:
        expected_mdr, expected_gst, expected_tds = expected_mdr_gst_tds_paise(gross_amount_paise, rail)
        result["expected_mdr_paise"] = expected_mdr
        result["expected_gst_on_mdr_paise"] = expected_gst
        result["expected_tds_paise"] = expected_tds
        result["computation_note"] = (
            "These are the STANDARD expected values for this gross amount and rail -- not a "
            "diagnosis. Compare them yourself against the settlement's actual figures to find "
            "which line deviates and by how much."
        )
    return result


def check_settlement_timing(
    order_timestamp: str, rail: str, is_international: bool = False, settlement_date: str | None = None
) -> dict:
    """order_timestamp: ISO datetime string for the order. Returns the
    expected settlement window in business days (IST calendar,
    src/common/calendar.py).

    settlement_date (optional, ISO date string): when given, also returns
    the ACTUAL business-day gap between the order and that settlement date,
    computed via src.common.calendar.business_days_between (already tested
    in Layer 1/2) -- not a diagnosis of whether/how a cutoff was violated.
    Compare it yourself against expected_window_business_days to decide.
    """
    expected_window = _settlement_window_business_days(rail, is_international)
    result = {
        "expected_window_business_days": expected_window,
        "rail_cutoff_rules": (
            f"{rail} settles in T+{expected_window} business days "
            f"({'international' if is_international else 'domestic'}), IST calendar, "
            "excluding weekends and Indian bank holidays."
        ),
    }
    if settlement_date is not None:
        order_date = datetime.fromisoformat(order_timestamp).date()
        actual_settlement_date = date.fromisoformat(settlement_date)
        result["actual_gap_business_days"] = business_days_between(order_date, actual_settlement_date)
        result["computation_note"] = (
            "actual_gap_business_days is a computed fact (business days between the order and the "
            "settlement date), not a diagnosis. Compare it against expected_window_business_days "
            "yourself to decide whether and how a cutoff was violated."
        )
    return result


TOOL_REGISTRY = {
    "query_merchant_contract": query_merchant_contract,
    "get_tax_rules": get_tax_rules,
    "check_settlement_timing": check_settlement_timing,
}
