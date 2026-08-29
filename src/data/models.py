"""Pydantic schemas for the three raw record types plus the ground-truth
label schema. All monetary fields are Decimal, parsed from str -- never a
float literal (CLAUDE.md Sec.3, enforced below with a field_validator that
rejects float input outright).
"""
from datetime import date, datetime
from decimal import Decimal
from typing import Literal, Optional

from pydantic import BaseModel, field_validator, model_validator

from src.common.calendar import IST

PaymentMethod = Literal["upi", "credit_card", "debit_card", "netbanking", "amex"]

ExpectedResolution = Literal["fast_path", "agent_resolved", "honest_exception"]


def _reject_float(v: Decimal | str | float, info) -> Decimal:
    if isinstance(v, float):
        raise ValueError(
            f"{info.field_name} must be a Decimal parsed from str, not a float literal"
        )
    return Decimal(v)


class InternalOrder(BaseModel):
    order_id: str
    gross_amount: Decimal
    customer_id: str
    payment_method: PaymentMethod
    timestamp: datetime
    refund_amount: Optional[Decimal] = None

    @field_validator("gross_amount", "refund_amount", mode="before")
    @classmethod
    def reject_float_money(cls, v, info):
        if v is None:
            return v
        return _reject_float(v, info)

    @field_validator("gross_amount")
    @classmethod
    def gross_amount_non_negative(cls, v: Decimal) -> Decimal:
        if v < 0:
            raise ValueError("gross_amount must be non-negative")
        return v

    @field_validator("timestamp")
    @classmethod
    def ensure_ist(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=IST)
        return v.astimezone(IST)

    @model_validator(mode="after")
    def refund_amount_within_bounds(self) -> "InternalOrder":
        if self.refund_amount is not None:
            if self.refund_amount <= 0:
                raise ValueError("refund_amount must be strictly positive")
            if self.refund_amount >= self.gross_amount:
                raise ValueError("refund_amount must be less than gross_amount")
        return self


class GatewaySettlement(BaseModel):
    payment_id: str
    order_id: str
    gross_amount: Decimal
    payment_method: PaymentMethod
    mdr: Decimal
    gst_on_mdr: Decimal
    tds_194o: Decimal
    net_amount: Decimal
    utr: str
    settlement_date: date
    is_international: bool = False

    @field_validator("gross_amount", "mdr", "gst_on_mdr", "tds_194o", "net_amount", mode="before")
    @classmethod
    def reject_float_money(cls, v, info):
        return _reject_float(v, info)

    @field_validator("gross_amount", "mdr", "gst_on_mdr", "tds_194o", "net_amount")
    @classmethod
    def non_negative(cls, v: Decimal, info) -> Decimal:
        if v < 0:
            raise ValueError(f"{info.field_name} must be non-negative")
        return v

    @model_validator(mode="after")
    def net_amount_invariant(self) -> "GatewaySettlement":
        expected_net = self.gross_amount - self.mdr - self.gst_on_mdr - self.tds_194o
        if self.net_amount != expected_net:
            raise ValueError(
                f"net_amount {self.net_amount} != gross_amount - mdr - gst_on_mdr - tds_194o "
                f"({expected_net})"
            )
        return self

    @model_validator(mode="after")
    def upi_has_nil_mdr_and_gst(self) -> "GatewaySettlement":
        if self.payment_method == "upi":
            if self.mdr != 0 or self.gst_on_mdr != 0:
                raise ValueError("UPI P2M MDR is nil by regulation: mdr and gst_on_mdr must be 0")
        return self


class BankStatementLine(BaseModel):
    utr: str
    credited_amount: Decimal
    value_date: date
    narration: str

    @field_validator("credited_amount", mode="before")
    @classmethod
    def reject_float_money(cls, v, info):
        return _reject_float(v, info)

    @field_validator("credited_amount")
    @classmethod
    def non_negative(cls, v: Decimal) -> Decimal:
        if v < 0:
            raise ValueError("credited_amount must be non-negative")
        return v


class GroundTruthEntry(BaseModel):
    order_id: str
    category: str
    expected_resolution: ExpectedResolution
    expected_root_cause_code: Optional[str] = None
    expected_delta_paise: Optional[int] = None
    notes: str = ""

    @model_validator(mode="after")
    def root_cause_and_delta_consistent_with_resolution(self) -> "GroundTruthEntry":
        if self.expected_resolution == "agent_resolved":
            if self.expected_root_cause_code is None or self.expected_delta_paise is None:
                raise ValueError(
                    "agent_resolved entries must have both expected_root_cause_code "
                    "and expected_delta_paise populated"
                )
        else:
            if self.expected_root_cause_code is not None or self.expected_delta_paise is not None:
                raise ValueError(
                    f"{self.expected_resolution} entries must leave expected_root_cause_code "
                    "and expected_delta_paise null"
                )
        return self
