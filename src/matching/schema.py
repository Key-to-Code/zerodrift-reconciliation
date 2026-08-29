"""Polars schemas and DataFrame loaders for the Layer 2 fast-path cascade.

Money is stored as integer paise everywhere in these frames -- never let
Polars infer a money column, or it silently coerces to Float64 (CLAUDE.md
Sec.3). Conversion happens only via src.common.money.to_paise, at the
boundary between the Pydantic-validated Decimal records and these frames.

order_id/utr stay pl.Utf8 for exact string equality -- no premature
categorical coercion on identifier columns (docs/plan.md Layer 2 Sec.2.1).
"""
from __future__ import annotations

import polars as pl

from src.common.money import to_paise
from src.data.models import BankStatementLine, GatewaySettlement, InternalOrder

orders_schema = {
    "order_id": pl.Utf8,
    "gross_amount_paise": pl.Int64,
    "customer_id": pl.Utf8,
    "payment_method": pl.Categorical,
    "timestamp": pl.Datetime("us"),
    # NEW vs. docs/plan.md Sec.2.1's literal example -- InternalOrder gained
    # refund_amount in Layer 1 Addendum A2, and the fast path needs it (see
    # src/matching/fast_path.py) to exclude refund_clawback from auto-match.
    "refund_amount_paise": pl.Int64,
}

settlement_schema = {
    "payment_id": pl.Utf8,
    "order_id": pl.Utf8,
    "gross_amount_paise": pl.Int64,
    # NEW vs. the literal example -- needed to recompute the expected
    # standard-rate MDR per rail in fast_path.py's revalidation step.
    "payment_method": pl.Categorical,
    "mdr_paise": pl.Int64,
    "gst_on_mdr_paise": pl.Int64,
    "tds_paise": pl.Int64,
    "net_amount_paise": pl.Int64,
    "utr": pl.Utf8,
    "settlement_date": pl.Date,
    # NEW vs. the literal example -- needed for the INTL_MARKUP settlement
    # window (T+3) in the timing revalidation step (Layer 1 Addendum A3).
    "is_international": pl.Boolean,
}

bank_schema = {
    "utr": pl.Utf8,
    "credited_amount_paise": pl.Int64,
    "value_date": pl.Date,
    "narration": pl.Utf8,
}


def orders_to_frame(orders: list[InternalOrder]) -> pl.DataFrame:
    rows = [
        {
            "order_id": o.order_id,
            "gross_amount_paise": to_paise(o.gross_amount),
            "customer_id": o.customer_id,
            "payment_method": o.payment_method,
            "timestamp": o.timestamp.replace(tzinfo=None),
            "refund_amount_paise": to_paise(o.refund_amount) if o.refund_amount is not None else None,
        }
        for o in orders
    ]
    return pl.DataFrame(rows, schema=orders_schema)


def settlements_to_frame(settlements: list[GatewaySettlement]) -> pl.DataFrame:
    rows = [
        {
            "payment_id": s.payment_id,
            "order_id": s.order_id,
            "gross_amount_paise": to_paise(s.gross_amount),
            "payment_method": s.payment_method,
            "mdr_paise": to_paise(s.mdr),
            "gst_on_mdr_paise": to_paise(s.gst_on_mdr),
            "tds_paise": to_paise(s.tds_194o),
            "net_amount_paise": to_paise(s.net_amount),
            "utr": s.utr,
            "settlement_date": s.settlement_date,
            "is_international": s.is_international,
        }
        for s in settlements
    ]
    return pl.DataFrame(rows, schema=settlement_schema)


def bank_lines_to_frame(bank_lines: list[BankStatementLine]) -> pl.DataFrame:
    rows = [
        {
            "utr": b.utr,
            "credited_amount_paise": to_paise(b.credited_amount),
            "value_date": b.value_date,
            "narration": b.narration,
        }
        for b in bank_lines
    ]
    return pl.DataFrame(rows, schema=bank_schema)
