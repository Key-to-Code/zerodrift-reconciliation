"""Structured resolution schema the agent must return (docs/plan.md Sec.4.1).

This is what makes a diagnosis gradeable: Layer 8's evaluate.py diffs
root_cause_code and quantified_delta_paise against ground_truth.json's
expected_root_cause_code / expected_delta_paise fields. A resolution that
avoids UNRESOLVED but gets the wrong code or delta is scored incorrect, not
merely "resolved."
"""
from typing import Literal

from pydantic import BaseModel, Field

RootCauseCode = Literal[
    "AMEX_SURCHARGE",
    "INTL_MARKUP",
    "MISSING_GST",
    "MISSING_TDS",
    "CUTOFF_T1",
    "CUTOFF_T2",
    "BATCH_LEVEL_FEE",
    "REFUND_NO_MDR_REVERSAL",
    "UNRESOLVED",
]


class AgentResolution(BaseModel):
    root_cause_code: RootCauseCode
    quantified_delta_paise: int
    evidence_tool_calls: list[str] = Field(default_factory=list)
    confidence_note: str
