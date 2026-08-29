"""IST business-day calendar. Excludes Saturday/Sunday and a small hardcoded
list of Indian bank holidays covering the frozen dataset's 2025 window (plus a
couple of dates outside that window used to prove the holiday list itself
works, per docs/plan.md Layer 1 Addendum A5). Shared by the generator's
cutoff_drift timing and Layer 2's date-window matching -- a raw calendar-day
window silently fails whenever it crosses a weekend, so business days are used
everywhere a settlement window is enforced.
"""
from datetime import date, timedelta
import zoneinfo

IST = zoneinfo.ZoneInfo("Asia/Kolkata")

INDIAN_BANK_HOLIDAYS: frozenset[date] = frozenset(
    {
        date(2025, 1, 26),  # Republic Day (falls on a Sunday)
        date(2025, 8, 15),  # Independence Day
        date(2025, 10, 2),  # Gandhi Jayanti
        date(2025, 10, 20),  # Diwali
    }
)


def is_business_day(d: date) -> bool:
    return d.weekday() < 5 and d not in INDIAN_BANK_HOLIDAYS


def add_business_days(d: date, n: int) -> date:
    if n == 0:
        return d
    step = 1 if n > 0 else -1
    remaining = abs(n)
    current = d
    while remaining > 0:
        current += timedelta(days=step)
        if is_business_day(current):
            remaining -= 1
    return current


def business_days_between(d1: date, d2: date) -> int:
    """Count of business days in the range (d1, d2] -- i.e. excluding d1,
    including d2. Negative if d2 precedes d1."""
    if d2 < d1:
        return -business_days_between(d2, d1)
    count = 0
    current = d1
    while current < d2:
        current += timedelta(days=1)
        if is_business_day(current):
            count += 1
    return count
