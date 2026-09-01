"""Layer 7: thin HTTP client wrapper around src/api/main.py's routes, used
only by src/dashboard/app.py. Streamlit never imports src.ledger,
src.matching, or src.forecast directly (docs/plan.md Layer 6/7's
modular-monolith transport boundary) -- every number the dashboard shows
travels through this module's HTTP calls to the FastAPI app.

Client injection: `_client_factory` is a module-level seam so tests can
point the dashboard at an in-process httpx.Client(app=fastapi_app) (or
FastAPI's own TestClient, which subclasses httpx.Client) instead of a real
base_url, without mocking any business logic -- only the transport target
changes. Streamlit re-execs app.py's *script* on every rerun, but the
api_client MODULE OBJECT is cached in sys.modules once imported, so a
monkeypatch made via set_client_factory() before AppTest.run() stays visible
across reruns.
"""
from __future__ import annotations

import os
from datetime import date
from typing import Callable

import httpx

DEFAULT_BASE_URL = os.environ.get("DASHBOARD_API_BASE_URL", "http://localhost:8000")


def _default_client_factory() -> httpx.Client:
    return httpx.Client(base_url=DEFAULT_BASE_URL, timeout=30.0)


_client_factory: Callable[[], httpx.Client] = _default_client_factory


def set_client_factory(factory: Callable[[], httpx.Client]) -> None:
    global _client_factory
    _client_factory = factory


def get_client() -> httpx.Client:
    return _client_factory()


class ApiClientError(Exception):
    """A clean, dashboard-displayable wrapper around an API error response
    (404 unknown batch_run_id, 422 validation) -- app.py catches this and
    shows st.error() instead of letting a raw exception crash the page."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"API error {status_code}: {detail}")


def _raise_for_status(resp: httpx.Response) -> None:
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except Exception:
            detail = resp.text
        raise ApiClientError(resp.status_code, str(detail))


def trigger_batch_run(source: str, seed: int | None = None, records: int = 100) -> dict:
    resp = get_client().post("/batch-runs", json={"source": source, "seed": seed, "records": records})
    _raise_for_status(resp)
    return resp.json()


def get_status(batch_run_id: str) -> dict:
    resp = get_client().get(f"/batch-runs/{batch_run_id}/status")
    _raise_for_status(resp)
    return resp.json()


def get_exceptions(batch_run_id: str) -> list[dict]:
    resp = get_client().get(f"/batch-runs/{batch_run_id}/exceptions")
    _raise_for_status(resp)
    return resp.json()


def get_trial_balance(batch_run_id: str) -> list[dict]:
    resp = get_client().get(f"/batch-runs/{batch_run_id}/trial-balance")
    _raise_for_status(resp)
    return resp.json()


def get_forecast(batch_run_id: str, as_of: date, horizon_days: int = 7) -> list[dict]:
    resp = get_client().get(
        f"/batch-runs/{batch_run_id}/forecast",
        params={"as_of": as_of.isoformat(), "horizon_days": horizon_days},
    )
    _raise_for_status(resp)
    return resp.json()


def parse_confidence_note_category(confidence_note: str | None) -> str:
    """Extracts the `discrepancy_reason=...` category from a
    `confidence_note` string (see src/orchestration/batch_runner.py's
    `discrepancy_reason=...; root_cause=...; ...` format). Presentation-only
    parsing of an existing, already-tested string -- no new business logic.
    Falls back to "unknown" for a note that doesn't follow the format (e.g.
    None, or a plain stub note used in some tests) rather than raising and
    crashing the exception table."""
    if not confidence_note:
        return "unknown"
    for part in confidence_note.split(";"):
        part = part.strip()
        if part.startswith("discrepancy_reason="):
            return part[len("discrepancy_reason="):] or "unknown"
    return "unknown"


def build_forecast_chart_data(forecast_rows: list[dict], as_of: date) -> dict[str, dict[str, int]]:
    """Buckets forecast rows by date into confirmed vs. projected totals
    (paise), filtered to within_horizon, for the confirmed-vs-projected
    forecast chart (docs/plan.md Layer 7). Confirmed rows carry no
    expected_cash_date (src/forecast/cashflow.py) -- they are bucketed under
    `as_of` itself, since that is the date the cash is already available.
    Returns {date_iso: {"confirmed": paise, "projected": paise}}, sorted by
    date."""
    buckets: dict[str, dict[str, int]] = {}
    for row in forecast_rows:
        if not row.get("within_horizon", False):
            continue
        status = row["account_status"]
        date_key = row["expected_cash_date"] or as_of.isoformat()
        bucket = buckets.setdefault(date_key, {"confirmed": 0, "projected": 0})
        bucket[status] = bucket.get(status, 0) + row["amount_paise"]
    return dict(sorted(buckets.items()))
