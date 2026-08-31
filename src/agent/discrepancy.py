"""DiscrepancyRecord: the unit of work handed to the Layer 4 agent.

Two source queues feed into the same unified DiscrepancyRecord shape:

1. build_unmatched_bank_line_queue() -- bank credits whose UTR never appears
   in any settlement at all (orphan, adversarial_trap). Layer 2's
   fast_path.py only ever iterates settlements_df to build UTR groups
   (src/matching/fast_path.py:190-196), so these bank lines never enter
   that loop and never reach FastPathResult.discrepancies. This module
   closes that gap with its own scan over bank_df, rather than modifying
   the already-tested fast_path.py.

2. build_settlement_discrepancy_queue() -- settled UTRs that fast_path.py's
   hop-1 rate/timing/refund revalidation rejected (fee_drift,
   missing_tax_line, cutoff_drift, refund_clawback), or that failed hop 3
   (short_settlement: zero bank candidates; duplicate_credit: 2+ ambiguous
   candidates). These come from run_fast_path(...).discrepancies. Per
   fast_path.py's own docstring (Layer 1 Addendum A1), the generator keeps
   GatewaySettlement.net_amount internally consistent with a real bank
   credit even for these anomaly categories -- fast_path.py just never
   looks for it, because the group is excluded at hop 1 before hop 3 runs.
   This builder reuses fast_path._find_candidates() (already-tested Layer 2
   logic) to actually retrieve it, rather than re-deriving narration
   matching. Reusing it also means duplicate_credit's two ambiguous bank
   lines fall out naturally -- _find_candidates() returns every line that
   matches the UTR/amount via the same phase1/phase2 matching a human
   reviewer would use, so it returns 1 candidate for the normal anomaly
   categories and 2 for duplicate_credit, with no special-casing needed.

find_candidate_orders() is a plain amount/date proximity search, independent
of ground_truth.json (the agent must never see ground-truth labels or notes
-- doing so would leak the answer). It exists so an orphan/adversarial_trap
DiscrepancyRecord carries a genuine temptation to force-match, the same way
a human reviewer skimming a bank statement would notice "this is
suspiciously close to ORD1045." Whether the agent resists that temptation
is exactly what the adversarial_trap guardrail test checks.

Deliberate: the search compares against settlements_df's net_amount_paise
and settlement_date, not orders_df's gross_amount_paise/timestamp. A bank
credit is a net settlement figure, so a gross-vs-net comparison would miss
the actual proximity almost every time (MDR/GST/TDS deductions plus the
settlement-window gap make gross order amount/date a different number and
date range from what lands in the bank account). src/data/generator.py
confirms this is exactly how adversarial_trap decoys are constructed --
jittered +/-200 paise and +/-1 day around a real twin order's *settlement*
net_amount and settlement_date, not the order itself.
"""
from __future__ import annotations

from datetime import date

import polars as pl
from pydantic import BaseModel

from src.matching.fast_path import _find_candidates, run_fast_path

AMOUNT_TOLERANCE_PAISE = 500  # +/- INR 5: a bit wider than the generator's known
DATE_TOLERANCE_DAYS = 2       # +/-200 paise / +/-1 day jitter, since a real system
                               # doesn't get to know the generator's exact bounds.


class BatchContext(BaseModel):
    parent_utr: str
    batch_size: int
    sibling_order_ids: list[str]
    aggregate_bank_credit_paise: int


class CandidateOrder(BaseModel):
    order_id: str
    settlement_net_amount_paise: int
    settlement_date: str
    payment_method: str


class OrderContext(BaseModel):
    order_id: str
    gross_amount_paise: int
    payment_method: str
    timestamp: str
    refund_amount_paise: int | None


class SettlementContext(BaseModel):
    mdr_paise: int
    gst_on_mdr_paise: int
    tds_paise: int
    net_amount_paise: int
    utr: str
    settlement_date: str
    is_international: bool


class BankCredit(BaseModel):
    utr: str
    credited_amount_paise: int
    value_date: str
    narration: str


class DiscrepancyRecord(BaseModel):
    discrepancy_reason: str  # fast_path.DiscrepancyItem.reason, or "unmatched_bank_line"
    order_context: OrderContext | None = None
    settlement_context: SettlementContext | None = None
    bank_credits: list[BankCredit] = []
    candidate_orders: list[CandidateOrder] = []
    batch_context: BatchContext | None = None


def find_candidate_orders(
    settlements_df: pl.DataFrame,
    bank_amount_paise: int,
    bank_value_date: date,
    amount_tolerance_paise: int = AMOUNT_TOLERANCE_PAISE,
    date_tolerance_days: int = DATE_TOLERANCE_DAYS,
) -> list[CandidateOrder]:
    lo, hi = bank_amount_paise - amount_tolerance_paise, bank_amount_paise + amount_tolerance_paise
    candidates: list[CandidateOrder] = []
    for row in settlements_df.iter_rows(named=True):
        net_paise = row["net_amount_paise"]
        settlement_date = row["settlement_date"]
        if lo <= net_paise <= hi and abs((settlement_date - bank_value_date).days) <= date_tolerance_days:
            candidates.append(
                CandidateOrder(
                    order_id=row["order_id"],
                    settlement_net_amount_paise=net_paise,
                    settlement_date=str(settlement_date),
                    payment_method=row["payment_method"],
                )
            )
    return candidates


def build_unmatched_bank_line_queue(
    orders_df: pl.DataFrame, settlements_df: pl.DataFrame, bank_df: pl.DataFrame
) -> list[DiscrepancyRecord]:
    """DiscrepancyRecords for bank credits whose UTR has no settlement at all
    (orphan, adversarial_trap). orders_df is accepted for interface symmetry
    with build_settlement_discrepancy_queue but is currently unused --
    candidates come from settlements, not orders (see module docstring).
    """
    settled_utrs = set(settlements_df["utr"].to_list())
    records: list[DiscrepancyRecord] = []
    for row in bank_df.iter_rows(named=True):
        if row["utr"] in settled_utrs:
            continue
        candidates = find_candidate_orders(settlements_df, row["credited_amount_paise"], row["value_date"])
        records.append(
            DiscrepancyRecord(
                discrepancy_reason="unmatched_bank_line",
                bank_credits=[
                    BankCredit(
                        utr=row["utr"],
                        credited_amount_paise=row["credited_amount_paise"],
                        value_date=str(row["value_date"]),
                        narration=row["narration"],
                    )
                ],
                candidate_orders=candidates,
            )
        )
    return records


def build_settlement_discrepancy_queue(
    orders_df: pl.DataFrame, settlements_df: pl.DataFrame, bank_df: pl.DataFrame
) -> list[DiscrepancyRecord]:
    """DiscrepancyRecords for settled UTRs that fast_path.py's fast path
    rejected: rate/timing/refund revalidation failures (fee_drift,
    missing_tax_line, cutoff_drift, refund_clawback), zero bank candidates
    (short_settlement), or ambiguous multiple candidates (duplicate_credit).
    """
    fast_path_result = run_fast_path(orders_df, settlements_df, bank_df)
    orders_by_id = {row["order_id"]: row for row in orders_df.iter_rows(named=True)}
    settlements_by_order = {row["order_id"]: row for row in settlements_df.iter_rows(named=True)}
    bank_rows = list(bank_df.iter_rows(named=True))

    records: list[DiscrepancyRecord] = []
    for item in fast_path_result.discrepancies:
        if item.reason == "no_matching_order":
            # Defensive only -- never occurs in the frozen dataset (every
            # settlement references a real order); skip rather than crash
            # the queue build on data that doesn't exist here.
            continue

        order_id = item.order_ids[0]
        order = orders_by_id[order_id]
        settlement = settlements_by_order[order_id]

        order_context = OrderContext(
            order_id=order_id,
            gross_amount_paise=order["gross_amount_paise"],
            payment_method=order["payment_method"],
            timestamp=str(order["timestamp"]),
            refund_amount_paise=order["refund_amount_paise"],
        )
        settlement_context = SettlementContext(
            mdr_paise=settlement["mdr_paise"],
            gst_on_mdr_paise=settlement["gst_on_mdr_paise"],
            tds_paise=settlement["tds_paise"],
            net_amount_paise=settlement["net_amount_paise"],
            utr=settlement["utr"],
            settlement_date=str(settlement["settlement_date"]),
            is_international=settlement["is_international"],
        )

        if item.reason == "no_bank_candidate":
            bank_credits: list[BankCredit] = []
        else:
            found = _find_candidates(
                bank_rows, settlement["utr"], settlement["net_amount_paise"], settlement["settlement_date"]
            )
            bank_credits = [
                BankCredit(
                    utr=b["utr"],
                    credited_amount_paise=b["credited_amount_paise"],
                    value_date=str(b["value_date"]),
                    narration=b["narration"],
                )
                for b, _method in found
            ]

        records.append(
            DiscrepancyRecord(
                discrepancy_reason=item.reason,
                order_context=order_context,
                settlement_context=settlement_context,
                bank_credits=bank_credits,
            )
        )
    return records
