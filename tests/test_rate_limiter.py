"""Tests for src/agent/rate_limiter.py -- the local daily-token-budget guard
added after a live debugging session found an uncaught groq.RateLimitError
(a daily-quota 429) propagating all the way to a bare 500 at the API layer
(src/api/main.py::trigger_batch_run). Pure unit tests, no network, no DB --
_DailyTokenTracker is plain in-process arithmetic and is_daily_quota_error
only inspects an exception's status_code/message.
"""
from __future__ import annotations

import httpx
import pytest
from groq import BadRequestError, RateLimitError

from src.agent.rate_limiter import (
    DAILY_TOKEN_BUDGET,
    RECOMMENDED_MAX_LIVE_SEED_RECORDS,
    SAFETY_MARGIN_TOKENS,
    AgentRateLimitedError,
    _DailyTokenTracker,
    check_budget_for_batch,
    daily_token_tracker,
    is_daily_quota_error,
)


def _rate_limit_error(message: str) -> RateLimitError:
    resp = httpx.Response(429, request=httpx.Request("POST", "http://x"), json={"error": {"message": message}})
    return RateLimitError(message, response=resp, body=None)


TPD_MESSAGE = (
    "Rate limit reached for model `openai/gpt-oss-120b` in organization `org_x` service tier "
    "`on_demand` on tokens per day (TPD): Limit 200000, Used 198791, Requested 2195. "
    "Please try again in 7m5.952s."
)
TPM_MESSAGE = (
    "Rate limit reached for model `openai/gpt-oss-120b` in organization `org_x` service tier "
    "`on_demand` on tokens per minute (TPM): Limit 8000, Used 7912, Requested 500. "
    "Please try again in 3.4s."
)


# ---------------------------------------------------------------------------
# is_daily_quota_error: distinguishes a DAILY (TPD) 429 from a routine
# PER-MINUTE (TPM) 429, and from a non-429/non-rate-limit error entirely.
# ---------------------------------------------------------------------------

def test_daily_quota_error_detected_from_tpd_message():
    assert is_daily_quota_error(_rate_limit_error(TPD_MESSAGE)) is True


def test_per_minute_error_not_treated_as_daily_quota():
    assert is_daily_quota_error(_rate_limit_error(TPM_MESSAGE)) is False


def test_non_rate_limit_error_not_treated_as_daily_quota():
    resp = httpx.Response(400, request=httpx.Request("POST", "http://x"), json={"error": {"message": "boom"}})
    err = BadRequestError("some unrelated 400", response=resp, body=None)
    assert is_daily_quota_error(err) is False


def test_plain_exception_not_treated_as_daily_quota():
    assert is_daily_quota_error(ValueError("not even an API error")) is False


# ---------------------------------------------------------------------------
# _DailyTokenTracker: real-usage accumulation, budget check, day rollover.
# A fresh instance per test -- never the module-level singleton, so these
# tests can't interact with (or be polluted by) a real live call elsewhere
# in the suite.
# ---------------------------------------------------------------------------

def test_tracker_starts_with_full_budget():
    tracker = _DailyTokenTracker()
    assert tracker.used_today() == 0
    assert tracker.remaining() == DAILY_TOKEN_BUDGET


def test_tracker_accumulates_real_usage():
    tracker = _DailyTokenTracker()
    tracker.record_usage(1000)
    tracker.record_usage(2500)
    assert tracker.used_today() == 3500
    assert tracker.remaining() == DAILY_TOKEN_BUDGET - 3500


def test_tracker_ignores_non_positive_usage():
    tracker = _DailyTokenTracker()
    tracker.record_usage(0)
    tracker.record_usage(-5)
    assert tracker.used_today() == 0


def test_check_budget_passes_when_well_under_cap():
    tracker = _DailyTokenTracker()
    tracker.record_usage(1000)
    tracker.check_budget()  # must not raise


def test_check_budget_raises_within_safety_margin():
    tracker = _DailyTokenTracker()
    tracker.record_usage(DAILY_TOKEN_BUDGET - SAFETY_MARGIN_TOKENS)
    with pytest.raises(AgentRateLimitedError):
        tracker.check_budget()


def test_check_budget_raises_once_over_cap():
    tracker = _DailyTokenTracker()
    tracker.record_usage(DAILY_TOKEN_BUDGET + 1)
    with pytest.raises(AgentRateLimitedError):
        tracker.check_budget()


def test_check_budget_error_message_names_the_frozen_dataset_fallback():
    """The whole point of this exception is to be actionable, not just
    typed -- the caller-facing message must point at the always-available
    escape hatch (CLAUDE.md's honesty-over-fabrication spirit applied to the
    live product's own error surface, not just to code correctness)."""
    tracker = _DailyTokenTracker()
    tracker.record_usage(DAILY_TOKEN_BUDGET)
    with pytest.raises(AgentRateLimitedError, match="frozen"):
        tracker.check_budget()


def test_tracker_resets_on_new_day(monkeypatch):
    import datetime as dt_module

    tracker = _DailyTokenTracker()
    tracker.record_usage(5000)
    assert tracker.used_today() == 5000

    real_datetime = dt_module.datetime

    class _TomorrowDatetime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            return real_datetime.now(tz) + dt_module.timedelta(days=1)

    monkeypatch.setattr("src.agent.rate_limiter.datetime", _TomorrowDatetime)
    assert tracker.used_today() == 0
    assert tracker.remaining() == DAILY_TOKEN_BUDGET


def test_reset_for_testing_clears_usage():
    tracker = _DailyTokenTracker()
    tracker.record_usage(5000)
    tracker.reset_for_testing()
    assert tracker.used_today() == 0


# ---------------------------------------------------------------------------
# check_budget_for_batch: the pre-flight, whole-batch check
# (src/orchestration/batch_runner.py's addendum, 2026-09-03) -- operates on
# the REAL module-level daily_token_tracker singleton, since that's what
# run_batch actually consults; each test resets it in a finally block so
# these can't leak state into any other test in the same pytest session.
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_shared_tracker():
    daily_token_tracker.reset_for_testing()
    yield
    daily_token_tracker.reset_for_testing()


def test_check_budget_for_batch_zero_needed_never_raises_even_when_exhausted():
    daily_token_tracker.record_usage(DAILY_TOKEN_BUDGET)  # fully exhausted
    check_budget_for_batch(n_live_calls_needed=0, avg_tokens_per_call=7000.0)  # must not raise


def test_check_budget_for_batch_passes_when_estimate_fits():
    daily_token_tracker.record_usage(1000)
    check_budget_for_batch(n_live_calls_needed=3, avg_tokens_per_call=7000.0)  # ~21000, plenty remains


def test_check_budget_for_batch_raises_when_estimate_exceeds_remaining():
    daily_token_tracker.record_usage(DAILY_TOKEN_BUDGET - 5000)  # only 5000 left
    with pytest.raises(AgentRateLimitedError, match="estimated"):
        check_budget_for_batch(n_live_calls_needed=3, avg_tokens_per_call=7000.0)  # needs ~21000


def test_check_budget_for_batch_error_names_the_frozen_dataset_fallback():
    daily_token_tracker.record_usage(DAILY_TOKEN_BUDGET)
    with pytest.raises(AgentRateLimitedError, match="frozen"):
        check_budget_for_batch(n_live_calls_needed=1, avg_tokens_per_call=7000.0)


# ---------------------------------------------------------------------------
# RECOMMENDED_MAX_LIVE_SEED_RECORDS regression guard (added 2026-09-04):
# re-derives the real estimate at this constant's own value against a fresh
# (never-cached) generated batch and the REAL measured average cost, so a
# future change to category-mix proportions or real per-record cost that
# would make this constant unsafe fails a test rather than silently
# stranding the README/dashboard's documented guidance.
# ---------------------------------------------------------------------------

def test_recommended_max_live_seed_records_fits_within_usable_budget():
    from pathlib import Path

    from src.agent.discrepancy import build_settlement_discrepancy_queue, build_unmatched_bank_line_queue
    from src.agent.run_log import average_real_tokens_per_live_call
    from src.data.generator import generate_batch
    from src.matching.schema import bank_lines_to_frame, orders_to_frame, settlements_to_frame

    project_root = Path(__file__).resolve().parents[1]
    batch = generate_batch(num_records=RECOMMENDED_MAX_LIVE_SEED_RECORDS, seed=31415926)
    orders_df = orders_to_frame(batch.orders)
    settlements_df = settlements_to_frame(batch.settlements)
    bank_df = bank_lines_to_frame(batch.bank_lines)
    needing_diagnosis = len(build_settlement_discrepancy_queue(orders_df, settlements_df, bank_df)) + len(
        build_unmatched_bank_line_queue(orders_df, settlements_df, bank_df)
    )
    avg = average_real_tokens_per_live_call(
        [project_root / "data" / "agent_runs" / "frozen_1.jsonl", project_root / "data" / "agent_runs" / "layer4_test_cache.jsonl"]
    )
    estimated = needing_diagnosis * avg
    usable_budget = DAILY_TOKEN_BUDGET - SAFETY_MARGIN_TOKENS
    assert estimated <= usable_budget, (
        f"RECOMMENDED_MAX_LIVE_SEED_RECORDS={RECOMMENDED_MAX_LIVE_SEED_RECORDS} no longer fits the usable "
        f"daily budget: {needing_diagnosis} records needing diagnosis x ~{avg:.0f} real avg tokens/record "
        f"= ~{estimated:.0f}, budget is {usable_budget}. Lower the constant in src/agent/rate_limiter.py "
        "and update the README/dashboard copy that cites it."
    )
