"""Layer 2: the 3-hop fast-path matching cascade over the Polars-typed frames
from src.matching.schema.

Two design decisions were resolved against the user before implementation
(see the Layer 2 planning conversation; both are binding, not incidental):

1. **Fast path includes rate/timing/refund revalidation, not just linking.**
   cutoff_drift, fee_drift, missing_tax_line and refund_clawback are all
   internally self-consistent between GatewaySettlement and BankStatementLine
   (the generator recomputes net_amount so they always tie out exactly --
   see docs/plan.md Layer 1 Addendum A1). A pure order<->payment<->UTR<->bank
   linking cascade would therefore auto-resolve all of them, which
   contradicts their `expected_resolution: agent_resolved` ground-truth label
   and the plan's 60-70% fast-path target (63/100 = clean_match+utr_batch
   only). So before a settlement group is even offered to hop 3, every
   member settlement must independently reconcile against (a) the standard
   MDR/GST(18%)/TDS(0.1%) rate table the generator itself uses, (b) the
   expected settlement window for its rail under the business-day calendar,
   and (c) carry no refund_amount on its order. Any deviation on any member
   excludes the whole group from the fast path.

2. **Hop 3 recovers the UTR from `narration` only.** The structured
   `bank_line.utr` column is never read as a matching key here -- one of the
   five narration templates the generator uses (the IMPS one) is deliberately
   truncated to the UTR's last 6 digits with no "UTR" prefix, specifically to
   force the phase-2 fuzzy fallback. Reading `bank_line.utr` directly would
   make that whole cascade inert.

Phase-2 fuzzy detail worth recording: comparing the *whole* narration string
against the *whole* UTR with rapidfuzz gives a low score for the truncated
template (boilerplate like "IMPS/RZPY/" and "/xx" dilutes it well under the
85 threshold). Isolating a loose alphanumeric candidate token from the
narration first, then fuzzy-comparing just that token against the UTR, is
what actually clears 85 -- the truncated 6-digit suffix is a literal
substring of the full UTR, so an isolated comparison scores 100.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date

import polars as pl
from rapidfuzz import fuzz

from src.common.calendar import business_days_between
from src.common.money import from_paise, to_paise
from src.data.generator import _compute_settlement_fields, _settlement_window_business_days

DATE_WINDOW_BUSINESS_DAYS = 2
FUZZY_THRESHOLD = 85

_EXACT_TOKEN_PATTERNS = (re.compile(r"UTR[A-Z0-9]+"), re.compile(r"ORD[0-9]+"))
_LOOSE_TOKEN_PATTERN = re.compile(r"[A-Z0-9]{4,}")


def extract_exact_tokens(narration: str) -> set[str]:
    """Phase 1 candidates: identifiers with a recognizable UTR/ORD prefix."""
    tokens: set[str] = set()
    for pattern in _EXACT_TOKEN_PATTERNS:
        tokens.update(pattern.findall(narration))
    return tokens


def extract_loose_candidates(narration: str) -> set[str]:
    """Phase 2 candidates: any alphanumeric run of 4+ chars, prefix or not."""
    return set(_LOOSE_TOKEN_PATTERN.findall(narration)) or {narration}


def phase1_exact_match(narration: str, utr: str) -> bool:
    return utr in extract_exact_tokens(narration)


def phase2_fuzzy_match(
    narration: str,
    utr: str,
    bank_amount_paise: int,
    group_amount_paise: int,
    bank_value_date: date,
    group_settlement_date: date,
) -> bool:
    if bank_amount_paise != group_amount_paise:
        return False
    if abs(business_days_between(group_settlement_date, bank_value_date)) > DATE_WINDOW_BUSINESS_DAYS:
        return False
    best_ratio = max(fuzz.partial_ratio(candidate, utr) for candidate in extract_loose_candidates(narration))
    return best_ratio >= FUZZY_THRESHOLD


def expected_mdr_gst_tds_paise(gross_paise: int, rail: str) -> tuple[int, int, int]:
    """Recompute expected MDR/GST/TDS in exact integer paise, via the same
    Decimal, ROUND_HALF_UP arithmetic the generator itself uses (never
    float) -- CLAUDE.md Sec.3.
    """
    gross = from_paise(gross_paise)
    mdr, gst, tds, _net = _compute_settlement_fields(gross, rail)
    return to_paise(mdr), to_paise(gst), to_paise(tds)


def expected_settlement_window_days(rail: str, is_international: bool) -> int:
    return _settlement_window_business_days(rail, is_international)


def compute_expected_rate_frame(settlements_df: pl.DataFrame) -> pl.DataFrame:
    """One row per settlement: the expected MDR/GST/TDS paise values recomputed
    from the standard rate table. Explicit Int64 schema throughout -- this
    frame is the thing under test for "never Float64" (CLAUDE.md Sec.3).
    """
    rows = []
    for s in settlements_df.iter_rows(named=True):
        expected_mdr, expected_gst, expected_tds = expected_mdr_gst_tds_paise(
            s["gross_amount_paise"], s["payment_method"]
        )
        rows.append(
            {
                "order_id": s["order_id"],
                "expected_mdr_paise": expected_mdr,
                "expected_gst_paise": expected_gst,
                "expected_tds_paise": expected_tds,
            }
        )
    return pl.DataFrame(
        rows,
        schema={
            "order_id": pl.Utf8,
            "expected_mdr_paise": pl.Int64,
            "expected_gst_paise": pl.Int64,
            "expected_tds_paise": pl.Int64,
        },
    )


@dataclass
class ResolvedGroup:
    utr: str
    order_ids: list[str]
    net_amount_paise: int
    bank_narration: str
    match_method: str  # "exact_token" | "fuzzy_fallback"


@dataclass
class DiscrepancyItem:
    reason: str
    order_ids: list[str] = field(default_factory=list)
    utr: str | None = None
    candidate_count: int = 0


@dataclass
class FastPathResult:
    resolved: list[ResolvedGroup]
    discrepancies: list[DiscrepancyItem]

    @property
    def resolved_order_ids(self) -> set[str]:
        return {oid for g in self.resolved for oid in g.order_ids}


def _find_candidates(
    bank_rows: list[dict], utr: str, group_amount_paise: int, settlement_date: date
) -> list[tuple[dict, str]]:
    candidates: list[tuple[dict, str]] = []
    for b_row in bank_rows:
        # Hop 2 gate: the aggregate settlement sum must reconcile against the
        # bank credit exactly before hop 3 even considers this bank line.
        if b_row["credited_amount_paise"] != group_amount_paise:
            continue
        narration = b_row["narration"]
        if phase1_exact_match(narration, utr):
            candidates.append((b_row, "exact_token"))
        elif phase2_fuzzy_match(
            narration, utr, b_row["credited_amount_paise"], group_amount_paise, b_row["value_date"], settlement_date
        ):
            candidates.append((b_row, "fuzzy_fallback"))
    return candidates


def run_fast_path(orders_df: pl.DataFrame, settlements_df: pl.DataFrame, bank_df: pl.DataFrame) -> FastPathResult:
    orders_by_id = {row["order_id"]: row for row in orders_df.iter_rows(named=True)}
    bank_rows = list(bank_df.iter_rows(named=True))
    expected_rate_by_order = {
        row["order_id"]: row for row in compute_expected_rate_frame(settlements_df).iter_rows(named=True)
    }

    resolved: list[ResolvedGroup] = []
    discrepancies: list[DiscrepancyItem] = []

    # Hop 1: order -> payment, exact key join on order_id.
    groups: dict[str, list[dict]] = {}
    for s_row in settlements_df.iter_rows(named=True):
        order = orders_by_id.get(s_row["order_id"])
        if order is None:
            discrepancies.append(DiscrepancyItem(reason="no_matching_order", order_ids=[s_row["order_id"]]))
            continue
        groups.setdefault(s_row["utr"], []).append({"settlement": s_row, "order": order})

    for utr, members in groups.items():
        order_ids = [m["order"]["order_id"] for m in members]

        clean = True
        for m in members:
            s, o = m["settlement"], m["order"]
            expected = expected_rate_by_order[s["order_id"]]
            rate_ok = (
                s["mdr_paise"] == expected["expected_mdr_paise"]
                and s["gst_on_mdr_paise"] == expected["expected_gst_paise"]
                and s["tds_paise"] == expected["expected_tds_paise"]
            )
            expected_window = expected_settlement_window_days(s["payment_method"], s["is_international"])
            actual_gap = business_days_between(o["timestamp"].date(), s["settlement_date"])
            timing_ok = actual_gap == expected_window
            refund_ok = o["refund_amount_paise"] is None
            if not (rate_ok and timing_ok and refund_ok):
                clean = False

        if not clean:
            discrepancies.append(
                DiscrepancyItem(reason="rate_or_timing_or_refund_deviation", order_ids=order_ids, utr=utr)
            )
            continue

        # Hop 2: aggregate settlements by UTR, exact integer sum.
        group_amount_paise = sum(m["settlement"]["net_amount_paise"] for m in members)
        settlement_date = members[0]["settlement"]["settlement_date"]

        # Hop 3: two-phase narration matching, plus the cardinality guardrail.
        candidates = _find_candidates(bank_rows, utr, group_amount_paise, settlement_date)

        if len(candidates) == 1:
            b_row, method = candidates[0]
            resolved.append(
                ResolvedGroup(
                    utr=utr,
                    order_ids=order_ids,
                    net_amount_paise=group_amount_paise,
                    bank_narration=b_row["narration"],
                    match_method=method,
                )
            )
        elif len(candidates) == 0:
            discrepancies.append(DiscrepancyItem(reason="no_bank_candidate", order_ids=order_ids, utr=utr))
        else:
            discrepancies.append(
                DiscrepancyItem(
                    reason="ambiguous_multiple_candidates",
                    order_ids=order_ids,
                    utr=utr,
                    candidate_count=len(candidates),
                )
            )

    return FastPathResult(resolved=resolved, discrepancies=discrepancies)
