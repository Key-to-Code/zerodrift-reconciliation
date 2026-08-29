"""The only two functions in the codebase allowed to convert between Decimal
rupees and integer paise. Every other module imports these rather than
reimplementing the conversion (CLAUDE.md Sec.3).
"""
from decimal import Decimal

PAISE_PER_RUPEE = 100


def to_paise(amount: Decimal) -> int:
    if not isinstance(amount, Decimal):
        raise TypeError(f"to_paise requires a Decimal, got {type(amount).__name__}")
    paise = amount * PAISE_PER_RUPEE
    if paise != paise.to_integral_value():
        raise ValueError(f"amount {amount} has sub-paise precision, cannot convert exactly")
    return int(paise)


def from_paise(paise: int) -> Decimal:
    if not isinstance(paise, int):
        raise TypeError(f"from_paise requires an int, got {type(paise).__name__}")
    return (Decimal(paise) / PAISE_PER_RUPEE).quantize(Decimal("0.01"))
