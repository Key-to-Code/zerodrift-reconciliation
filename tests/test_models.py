"""Layer 1 tests for src/data/models.py -- Pydantic model-level validators,
isolated from the generator (see docs/plan.md Layer 1 Addendum, this file was
proposed as a new test file not explicitly named in the original plan).
"""
from datetime import date, datetime
from decimal import Decimal

import pytest
from pydantic import ValidationError

from src.common.calendar import IST
from src.data.models import BankStatementLine, GatewaySettlement, InternalOrder


def make_order(**overrides):
    fields = dict(
        order_id="ORD0001",
        gross_amount="1000.00",
        customer_id="CUST0001",
        payment_method="credit_card",
        timestamp=datetime(2025, 1, 6, 10, 0, 0),
    )
    fields.update(overrides)
    return InternalOrder(**fields)


def make_settlement(**overrides):
    fields = dict(
        payment_id="PAY0001",
        order_id="ORD0001",
        gross_amount="1000.00",
        payment_method="credit_card",
        mdr="20.00",
        gst_on_mdr="3.60",
        tds_194o="1.00",
        net_amount="975.40",
        utr="UTR0001",
        settlement_date=date(2025, 1, 8),
    )
    fields.update(overrides)
    return GatewaySettlement(**fields)


# ---------------------------------------------------------------------------
# Float rejection (CLAUDE.md Sec.3, supports criteria 5 & 9)
# ---------------------------------------------------------------------------

def test_internal_order_rejects_float_gross_amount():
    with pytest.raises(ValidationError):
        make_order(gross_amount=1000.00)


def test_gateway_settlement_rejects_float_mdr():
    with pytest.raises(ValidationError):
        make_settlement(mdr=20.00)


def test_bank_statement_line_rejects_float_credited_amount():
    with pytest.raises(ValidationError):
        BankStatementLine(
            utr="UTR0001",
            credited_amount=975.40,
            value_date=date(2025, 1, 8),
            narration="NEFT-UTR0001-SETTLE",
        )


def test_internal_order_accepts_decimal_from_string():
    order = make_order(gross_amount="1000.00")
    assert order.gross_amount == Decimal("1000.00")


# ---------------------------------------------------------------------------
# net_amount exact invariant (criterion 9)
# ---------------------------------------------------------------------------

def test_settlement_net_amount_invariant_holds():
    settlement = make_settlement()
    assert settlement.net_amount == (
        settlement.gross_amount - settlement.mdr - settlement.gst_on_mdr - settlement.tds_194o
    )


def test_settlement_rejects_net_amount_mismatch():
    with pytest.raises(ValidationError):
        make_settlement(net_amount="975.41")  # off by one paisa


def test_settlement_rejects_negative_monetary_field():
    with pytest.raises(ValidationError):
        make_settlement(mdr="-20.00")


# ---------------------------------------------------------------------------
# UPI nil-MDR/GST constraint (criterion 5, CLAUDE.md Sec.4)
# ---------------------------------------------------------------------------

def test_upi_settlement_accepts_zero_mdr_and_gst():
    settlement = make_settlement(
        payment_method="upi",
        mdr="0.00",
        gst_on_mdr="0.00",
        tds_194o="1.00",
        net_amount="999.00",
    )
    assert settlement.payment_method == "upi"
    assert settlement.mdr == Decimal("0.00")
    assert settlement.gst_on_mdr == Decimal("0.00")


def test_upi_settlement_rejects_nonzero_mdr():
    with pytest.raises(ValidationError):
        make_settlement(
            payment_method="upi",
            mdr="20.00",
            gst_on_mdr="0.00",
            net_amount="979.00",
        )


def test_upi_settlement_rejects_nonzero_gst_on_mdr():
    with pytest.raises(ValidationError):
        make_settlement(
            payment_method="upi",
            mdr="0.00",
            gst_on_mdr="3.60",
            net_amount="995.40",
        )


# ---------------------------------------------------------------------------
# IST timestamp pinning (spec Sec.1.3)
# ---------------------------------------------------------------------------

def test_naive_timestamp_gets_ist_attached():
    order = make_order(timestamp=datetime(2025, 1, 6, 10, 0, 0))
    assert order.timestamp.tzinfo is not None
    assert order.timestamp.utcoffset().total_seconds() == 5.5 * 3600


def test_aware_non_ist_timestamp_gets_converted_to_ist():
    import zoneinfo

    utc_time = datetime(2025, 1, 6, 4, 30, 0, tzinfo=zoneinfo.ZoneInfo("UTC"))
    order = make_order(timestamp=utc_time)
    assert order.timestamp.tzinfo is not None
    assert order.timestamp.utcoffset().total_seconds() == 5.5 * 3600
    # 04:30 UTC == 10:00 IST
    assert order.timestamp.hour == 10
    assert order.timestamp.minute == 0


# ---------------------------------------------------------------------------
# refund_clawback field (docs/plan.md Layer 1 Addendum A2)
# ---------------------------------------------------------------------------

def test_internal_order_refund_amount_defaults_to_none():
    order = make_order()
    assert order.refund_amount is None


def test_internal_order_refund_amount_must_be_less_than_gross():
    with pytest.raises(ValidationError):
        make_order(gross_amount="1000.00", refund_amount="1000.00")


def test_internal_order_refund_amount_must_be_positive():
    with pytest.raises(ValidationError):
        make_order(gross_amount="1000.00", refund_amount="0.00")


def test_internal_order_accepts_valid_refund_amount():
    order = make_order(gross_amount="1000.00", refund_amount="400.00")
    assert order.refund_amount == Decimal("400.00")


# ---------------------------------------------------------------------------
# is_international field (docs/plan.md Layer 1 Addendum A3)
# ---------------------------------------------------------------------------

def test_settlement_is_international_defaults_to_false():
    settlement = make_settlement()
    assert settlement.is_international is False


def test_settlement_accepts_is_international_true():
    settlement = make_settlement(is_international=True)
    assert settlement.is_international is True
