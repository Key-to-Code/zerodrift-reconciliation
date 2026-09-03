"""Local, in-process guard against Groq's free-tier daily token cap.

Added after a live debugging session (2026-09-03) found two real problems
with the agent's only prior protection, _invoke_with_backoff in graph.py:

1. A DAILY-quota (TPD) 429 was retried by that backoff loop the same way as
   a routine PER-MINUTE (TPM) 429 -- but a TPD exhaustion does not clear in
   the few seconds that loop sleeps, so retrying just delays an identical
   failure by ~30s before raising anyway.
2. Nothing tracked cumulative spend locally, so the system always
   discovered it was out of budget by actually attempting a live call and
   getting a real 429 from Groq -- which then propagated, uncaught, through
   run_batch all the way to FastAPI's generic exception handler, surfacing
   as a bare "Internal Server Error" with the real reason hidden.

This module is a client-side safety margin, not the authoritative limiter
-- Groq's own API remains authoritative, and a 429 it returns is still
handled (distinguished TPD-vs-TPM, see is_daily_quota_error below). This
just avoids wasting a live call once OUR OWN counted usage already tells us
the answer, and gives the API layer (src/api/main.py) a typed exception to
catch and turn into a clear, honest response instead of an opaque 500.

DAILY_TOKEN_BUDGET mirrors Groq's own documented cap for this key/model,
observed directly in a live 429 body: "Limit 200000 ... tokens per day".
SAFETY_MARGIN_TOKENS leaves headroom for the next call's own cost, which
isn't known in advance -- Groq bills real usage, only visible after a call
returns (src/agent/graph.py's _response_tokens).
"""
from __future__ import annotations

import math
import threading
from datetime import datetime, timezone

DAILY_TOKEN_BUDGET = 200_000
SAFETY_MARGIN_TOKENS = 10_000


class AgentRateLimitedError(Exception):
    """Raised instead of attempting (or retrying) a live model call once
    either this process's own tracked usage today is within
    SAFETY_MARGIN_TOKENS of DAILY_TOKEN_BUDGET, or Groq itself has just
    reported a daily-quota 429. Deliberately NOT a subclass of groq's
    APIStatusError -- callers (src/api/main.py) can catch this without
    importing the groq package."""


class _DailyTokenTracker:
    """Tracks tokens used since UTC midnight. Real usage only (Groq's own
    usage_metadata on each response, same accounting graph.py's per-record
    tokens_used already uses) -- never an estimate. A calendar-day reset is
    an approximation of Groq's actual rolling window (their own 429 message
    says "try again in Nm" as usage ages out continuously, not at a fixed
    boundary) -- close enough for a client-side safety margin, since Groq's
    API is still the authoritative last word either way."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._day: str | None = None
        self._used = 0

    def _roll_if_new_day_locked(self) -> None:
        today = datetime.now(timezone.utc).date().isoformat()
        if self._day != today:
            self._day = today
            self._used = 0

    def record_usage(self, tokens: int) -> None:
        if tokens <= 0:
            return
        with self._lock:
            self._roll_if_new_day_locked()
            self._used += tokens

    def used_today(self) -> int:
        with self._lock:
            self._roll_if_new_day_locked()
            return self._used

    def remaining(self) -> int:
        return DAILY_TOKEN_BUDGET - self.used_today()

    def check_budget(self) -> None:
        """Raises AgentRateLimitedError, with zero network call, if today's
        tracked usage already leaves less than SAFETY_MARGIN_TOKENS."""
        remaining = self.remaining()
        if remaining <= SAFETY_MARGIN_TOKENS:
            raise AgentRateLimitedError(
                f"Local daily token budget nearly exhausted: {self.used_today()}/{DAILY_TOKEN_BUDGET} "
                f"used today, {remaining} remaining (safety margin {SAFETY_MARGIN_TOKENS}) -- refusing "
                "to attempt a new live call rather than risk a mid-record failure. Use source='frozen' "
                "(seed=42, records=100), which never calls the live model, or retry once Groq's daily "
                "window rolls over."
            )

    def reset_for_testing(self) -> None:
        with self._lock:
            self._day = None
            self._used = 0


daily_token_tracker = _DailyTokenTracker()


def check_budget_for_batch(n_live_calls_needed: int, avg_tokens_per_call: float) -> None:
    """Pre-flight check for an entire batch, called BEFORE any ledger
    posting begins -- not just before each individual live call (that's
    diagnose_discrepancy's own per-record check_budget, which by definition
    can only fire after some earlier records in the same batch have already
    been posted). Added after a live debugging session found that hitting
    the daily quota mid-seed-batch crashed with partial progress already
    committed -- there was no soft landing for a seed batch (unlike the
    frozen path's cache/replay safety net) once a live call became
    necessary.

    n_live_calls_needed=0 (e.g. the frozen dataset, always fully cached, or
    a seed recipe that happens to already be fully cached) never raises,
    regardless of remaining budget -- quota is simply irrelevant when zero
    live calls will actually be attempted.

    Deliberately does NOT estimate a wait time: Groq's window is rolling,
    not a fixed reset, and this process has no way to know how much of
    today's usage will age out by when -- inventing an ETA would violate
    CLAUDE.md Sec.1 (never state a number not actually derived). The
    message instead points at the real, always-available alternative.
    """
    if n_live_calls_needed <= 0:
        return
    estimated_tokens = math.ceil(n_live_calls_needed * avg_tokens_per_call)
    remaining = daily_token_tracker.remaining()
    if estimated_tokens > remaining:
        raise AgentRateLimitedError(
            f"This batch needs an estimated {estimated_tokens} tokens ({n_live_calls_needed} records "
            f"needing live agent diagnosis x ~{avg_tokens_per_call:.0f} tokens/record, the real measured "
            f"average) for its live agent phase, but only {remaining} remain in today's local budget "
            "tracking -- refusing to start rather than fail partway through with some records already "
            "posted. Use source='frozen' (seed=42, records=100), which needs zero live calls, reduce "
            "records, or retry once more of today's usage has aged out of Groq's rolling daily window."
        )


def is_daily_quota_error(exc: Exception) -> bool:
    """True for a Groq 429 whose message identifies it as a DAILY (TPD)
    quota exhaustion, as opposed to the routine PER-MINUTE (TPM) limit
    _invoke_with_backoff already retries successfully. Groq's client
    doesn't expose a structured TPD-vs-TPM field, only this message text,
    e.g.: 'Rate limit reached ... on tokens per day (TPD): Limit 200000 ...'
    """
    from groq import APIStatusError

    if not isinstance(exc, APIStatusError) or exc.status_code != 429:
        return False
    text = str(exc).lower()
    return "tokens per day" in text or "(tpd)" in text
