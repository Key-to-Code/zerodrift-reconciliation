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


def format_inr(amount: Decimal) -> str:
    """Formats a Decimal rupee amount with Indian digit grouping (2,2,3 --
    lakh/crore), e.g. Decimal("482300.00") -> "4,82,300.00", not the Western
    3-digit grouping Python's `:,` format spec gives. Presentation only --
    takes an already-`from_paise()`-derived Decimal, never touches paise or
    does money math itself."""
    if not isinstance(amount, Decimal):
        raise TypeError(f"format_inr requires a Decimal, got {type(amount).__name__}")
    sign = "-" if amount < 0 else ""
    quantized = abs(amount).quantize(Decimal("0.01"))
    int_part, _, frac_part = str(quantized).partition(".")
    if len(int_part) <= 3:
        grouped = int_part
    else:
        groups = [int_part[-3:]]
        rest = int_part[:-3]
        while len(rest) > 2:
            groups.insert(0, rest[-2:])
            rest = rest[:-2]
        if rest:
            groups.insert(0, rest)
        grouped = ",".join(groups)
    return f"{sign}{grouped}.{frac_part}"
