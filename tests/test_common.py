"""Layer 1 tests for src/common: money.py (paise round-trip) and calendar.py
(IST business-day calendar). Written before implementation per CLAUDE.md's build
protocol -- these must fail honestly until src/common/{money,calendar}.py exist.
"""
from datetime import date
from decimal import Decimal

import pytest

from src.common.money import from_paise, to_paise
from src.common.calendar import add_business_days, business_days_between, is_business_day


# ---------------------------------------------------------------------------
# money.py -- to_paise / from_paise round-trip (Layer 1 criterion 10a)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "amount",
    [
        Decimal("0.00"),
        Decimal("1.00"),
        Decimal("0.01"),
        Decimal("0.99"),
        Decimal("100.50"),
        Decimal("1234567.89"),
        Decimal("9999999999.99"),
        Decimal("42.42"),
    ],
)
def test_to_paise_from_paise_roundtrip_exact(amount):
    paise = to_paise(amount)
    assert isinstance(paise, int)
    assert from_paise(paise) == amount


def test_to_paise_is_exact_multiplication_by_100():
    assert to_paise(Decimal("100.50")) == 10050
    assert to_paise(Decimal("0.01")) == 1
    assert to_paise(Decimal("0.00")) == 0


def test_from_paise_is_exact_division_by_100():
    assert from_paise(10050) == Decimal("100.50")
    assert from_paise(1) == Decimal("0.01")
    assert from_paise(0) == Decimal("0.00")


def test_to_paise_rejects_float_input():
    with pytest.raises((TypeError, ValueError)):
        to_paise(100.50)  # type: ignore[arg-type]


def test_paise_roundtrip_zero_drift_over_many_values():
    # A broader sweep than the parametrized cases above -- simulates the
    # "full frozen dataset, zero drift" round-trip claim from CLAUDE.md Sec.3.
    amounts = [Decimal(f"{i}.{i % 100:02d}") for i in range(0, 5000, 37)]
    for amount in amounts:
        assert from_paise(to_paise(amount)) == amount


# ---------------------------------------------------------------------------
# calendar.py -- IST business-day calendar (Layer 1 criterion 10b)
# ---------------------------------------------------------------------------

def test_is_business_day_true_for_ordinary_weekday():
    # 2025-01-06 is a Monday, not a holiday.
    assert is_business_day(date(2025, 1, 6)) is True


def test_is_business_day_skips_saturday_and_sunday():
    # 2025-01-11 is a Saturday, 2025-01-12 is a Sunday.
    assert is_business_day(date(2025, 1, 11)) is False
    assert is_business_day(date(2025, 1, 12)) is False


def test_is_business_day_skips_hardcoded_holidays():
    # Independence Day 2025-08-15 falls on a Friday -- a holiday that
    # wouldn't already be excluded by the weekend check (docs/plan.md A5).
    assert is_business_day(date(2025, 8, 15)) is False
    # Gandhi Jayanti 2025-10-02.
    assert is_business_day(date(2025, 10, 2)) is False
    # Diwali 2025-10-20.
    assert is_business_day(date(2025, 10, 20)) is False


def test_is_business_day_holiday_on_a_weekend_does_not_error():
    # Republic Day 2025-01-26 falls on a Sunday -- already excluded by the
    # weekend rule; the holiday list overlapping it must not raise or double-count.
    assert is_business_day(date(2025, 1, 26)) is False


def test_add_business_days_skips_plain_weekend():
    # 2025-01-10 is a Friday. +1 business day should land on Monday 2025-01-13,
    # not Saturday.
    assert add_business_days(date(2025, 1, 10), 1) == date(2025, 1, 13)


def test_add_business_days_crosses_weekend_and_holiday():
    # 2025-08-14 is a Thursday. +1 business day should skip Friday 2025-08-15
    # (Independence Day) and the following weekend, landing on Monday 2025-08-18.
    assert add_business_days(date(2025, 8, 14), 1) == date(2025, 8, 18)


def test_add_business_days_zero_returns_same_day_if_business_day():
    assert add_business_days(date(2025, 1, 6), 0) == date(2025, 1, 6)


def test_add_business_days_multi_day_span():
    # 2025-01-06 is a Monday. +5 business days = Monday 2025-01-13.
    assert add_business_days(date(2025, 1, 6), 5) == date(2025, 1, 13)


def test_business_days_between_excludes_weekend():
    # Friday 2025-01-10 to Monday 2025-01-13 is 1 business day apart, not 3.
    assert business_days_between(date(2025, 1, 10), date(2025, 1, 13)) == 1


def test_business_days_between_same_day_is_zero():
    assert business_days_between(date(2025, 1, 6), date(2025, 1, 6)) == 0


def test_business_days_between_naive_calendar_day_window_would_be_wrong():
    # Regression test named per Layer 2 acceptance criteria: a raw +/-2
    # calendar-day window silently fails whenever it crosses a weekend.
    # Thursday 2025-01-09 to Monday 2025-01-13 is 4 raw calendar days apart,
    # which a naive "within 2 calendar days" check would reject -- but it is
    # only 2 business days apart, which a correct window must accept.
    start = date(2025, 1, 9)
    end = date(2025, 1, 13)
    raw_calendar_days = (end - start).days
    assert raw_calendar_days == 4
    assert business_days_between(start, end) == 2
