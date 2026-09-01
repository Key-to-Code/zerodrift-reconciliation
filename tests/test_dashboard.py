"""Layer 7 tests for the Streamlit dashboard (src/dashboard/app.py and
src/dashboard/api_client.py). Written before implementation per CLAUDE.md's
build protocol.

Two groups of test live here:

- api_client-level tests (1-13) exercise src/dashboard/api_client.py's HTTP
  calls and pure data-shaping helpers directly, against the real FastAPI app
  (src/api/main.py) via its own TestClient, wired in as api_client's client
  through set_client_factory(). This proves the dashboard's data layer is
  correct without needing a running Streamlit process.
- AppTest-level tests (14-21) drive src/dashboard/app.py itself via
  streamlit.testing.v1.AppTest -- Streamlit's headless test harness -- to
  prove the page actually wires those calls into widgets and renders
  without crashing. AppTest cannot inspect st.bar_chart's rendered content
  (confirmed by direct introspection of streamlit==1.62.0's AppTest class:
  no chart-data accessor exists), so the forecast-chart test only proves the
  chart section executes without raising -- a human should still glance at
  the real rendered chart once. Everything else (metrics, dataframes,
  alerts, exceptions) IS inspectable and is asserted on directly.

Runs against a real Postgres test database (tests/conftest.py) -- CLAUDE.md
forbids mocking the ledger layer, so these tests go through the real
API -> orchestration -> Postgres path, exactly like a browser would.

The `client` fixture below is intentionally duplicated from test_api.py
(same dependency-override pattern) rather than shared via conftest.py, per
CLAUDE.md's scope-discipline rule against touching a prior layer's test
file when a small, local duplication does the job.
"""
from __future__ import annotations

import uuid
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.dashboard import api_client
from src.ledger.models import get_sessionmaker

APP_PATH = str(Path(__file__).resolve().parents[1] / "src" / "dashboard" / "app.py")

FROZEN_FAST_PATH_COUNT = 63
FROZEN_AGENT_RESOLVED_COUNT = 20
FROZEN_HONEST_EXCEPTION_COUNT = 17
FROZEN_TOTAL_GROUND_TRUTH_ENTRIES = 100


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(pg_engine):
    from src.api.main import app, get_session

    session_factory = get_sessionmaker(pg_engine)

    def _override_get_session():
        session = session_factory()
        try:
            yield session
        finally:
            session.rollback()
            session.close()

    app.dependency_overrides[get_session] = _override_get_session
    try:
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


@pytest.fixture()
def dashboard_client(client):
    """Wires api_client's module-level client factory to the real FastAPI
    TestClient (which subclasses httpx.Client) instead of a live base_url --
    same dependency-injection seam AppTest.from_file's re-exec of app.py
    will see, since api_client is a cached module in sys.modules."""
    original_factory = api_client._client_factory
    api_client.set_client_factory(lambda: client)
    try:
        yield client
    finally:
        api_client.set_client_factory(original_factory)


# ---------------------------------------------------------------------------
# Test 1 -- app.py never imports the ledger/matching/forecast modules
# directly; it must only reach them through the HTTP API (AC8)
# ---------------------------------------------------------------------------

def test_app_module_does_not_import_ledger_matching_forecast_directly():
    import ast

    tree = ast.parse(Path(APP_PATH).read_text(encoding="utf-8"))
    forbidden_prefixes = ("src.ledger", "src.matching", "src.forecast", "src.orchestration")
    imported_modules = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    for module in imported_modules:
        for forbidden in forbidden_prefixes:
            assert not module.startswith(forbidden), (
                f"app.py must not import {module} directly -- use api_client instead"
            )


# ---------------------------------------------------------------------------
# Test 2 -- triggering a frozen run through api_client returns a usable
# summary, same shape the API itself returns (AC1)
# ---------------------------------------------------------------------------

def test_api_client_trigger_batch_run_frozen_returns_summary(dashboard_client):
    summary = api_client.trigger_batch_run("frozen")
    uuid.UUID(summary["batch_run_id"])  # must be a real uuid, not raise
    assert summary["fast_path_count"] == FROZEN_FAST_PATH_COUNT
    assert summary["agent_resolved_count"] == FROZEN_AGENT_RESOLVED_COUNT
    assert summary["honest_exception_count"] == FROZEN_HONEST_EXCEPTION_COUNT


# ---------------------------------------------------------------------------
# Test 3 -- triggering a seed run without a seed surfaces a clean
# ApiClientError (422), not a raw httpx/requests exception (AC1, AC8)
# ---------------------------------------------------------------------------

def test_api_client_trigger_batch_run_seed_without_seed_raises_clean_error(dashboard_client):
    with pytest.raises(api_client.ApiClientError) as exc_info:
        api_client.trigger_batch_run("seed", seed=None)
    assert exc_info.value.status_code == 422


# ---------------------------------------------------------------------------
# Test 4 -- an unknown batch_run_id surfaces a clean 404 ApiClientError, the
# exact error the manual "add existing run" box must display (AC2, AC8)
# ---------------------------------------------------------------------------

def test_api_client_get_status_unknown_run_raises_404_error(dashboard_client):
    with pytest.raises(api_client.ApiClientError) as exc_info:
        api_client.get_status(str(uuid.uuid4()))
    assert exc_info.value.status_code == 404


# ---------------------------------------------------------------------------
# Test 5 -- status counts for the frozen run match the already-proven
# Layer 6 figures (AC3)
# ---------------------------------------------------------------------------

def test_api_client_get_status_matches_known_frozen_counts(dashboard_client):
    batch_run_id = api_client.trigger_batch_run("frozen")["batch_run_id"]
    status = api_client.get_status(batch_run_id)
    assert status["fast_path"] == FROZEN_FAST_PATH_COUNT
    assert status["agent_resolved"] == FROZEN_AGENT_RESOLVED_COUNT
    assert status["honest_exception"] == FROZEN_HONEST_EXCEPTION_COUNT
    assert status["total"] == FROZEN_TOTAL_GROUND_TRUTH_ENTRIES


# ---------------------------------------------------------------------------
# Test 6 -- exception list has 17 entries, each with a non-blank
# confidence_note (the raw material the category column parses) (AC4)
# ---------------------------------------------------------------------------

def test_api_client_get_exceptions_returns_17_with_notes(dashboard_client):
    batch_run_id = api_client.trigger_batch_run("frozen")["batch_run_id"]
    exceptions = api_client.get_exceptions(batch_run_id)
    assert len(exceptions) == FROZEN_HONEST_EXCEPTION_COUNT
    for row in exceptions:
        assert row["confidence_note"]


# ---------------------------------------------------------------------------
# Test 7 -- category parsing extracts discrepancy_reason from a real
# batch_runner.py-formatted note (AC4)
# ---------------------------------------------------------------------------

def test_parse_confidence_note_category_extracts_discrepancy_reason():
    note = "discrepancy_reason=no_bank_candidate; root_cause=UNRESOLVED; delta_paise=0; stub note text"
    assert api_client.parse_confidence_note_category(note) == "no_bank_candidate"


# ---------------------------------------------------------------------------
# Test 8 -- category parsing falls back to "unknown" for a malformed or
# missing note, rather than raising and breaking the exception table (AC4)
# ---------------------------------------------------------------------------

def test_parse_confidence_note_category_handles_malformed_note():
    assert api_client.parse_confidence_note_category(None) == "unknown"
    assert api_client.parse_confidence_note_category("") == "unknown"
    assert api_client.parse_confidence_note_category("stub note with no fields") == "unknown"


# ---------------------------------------------------------------------------
# Test 9 -- trial balance served through api_client includes the TOTAL row
# and it nets to exactly zero for a fully-settled run (AC5)
# ---------------------------------------------------------------------------

def test_api_client_get_trial_balance_includes_zero_total_row(dashboard_client):
    batch_run_id = api_client.trigger_batch_run("frozen")["batch_run_id"]
    rows = api_client.get_trial_balance(batch_run_id)
    total_row = [r for r in rows if r["account_code"] == "TOTAL"][0]
    assert total_row["net_balance_paise"] == 0
    assert len(rows) > 1  # a real per-account table, not just the total


# ---------------------------------------------------------------------------
# Test 10 -- forecast rows carry the confirmed/projected + within_horizon
# fields the chart needs (AC6)
# ---------------------------------------------------------------------------

def test_api_client_get_forecast_returns_rows_with_valid_status(dashboard_client):
    # A frozen run posts Stage 1 + Stage 2 for the whole batch synchronously
    # in one run_batch() call (src/orchestration/batch_runner.py), so by the
    # time this HTTP call returns, every in-flight order is already
    # "confirmed" -- "projected" only exists for a ledger state where Stage 2
    # (or a utr_batch's sibling bank-credit entry) has not yet posted, which
    # this fully-synchronous pipeline never leaves observable. Layer 5's own
    # tests (test_forecast.py) already cover the "projected" branch directly
    # against a hand-built partial ledger state -- this test only needs to
    # prove the dashboard's HTTP path returns real, validly-shaped rows.
    batch_run_id = api_client.trigger_batch_run("frozen")["batch_run_id"]
    rows = api_client.get_forecast(batch_run_id, as_of=date(2025, 1, 6), horizon_days=7)
    assert rows
    statuses = {row["account_status"] for row in rows}
    assert statuses <= {"confirmed", "projected"}
    assert any(row["account_status"] == "confirmed" for row in rows)


# ---------------------------------------------------------------------------
# Test 11 -- chart data-shaping buckets confirmed vs. projected amounts by
# date, confirmed rows landing on as_of (AC6)
# ---------------------------------------------------------------------------

def test_build_forecast_chart_data_buckets_confirmed_vs_projected_by_date():
    as_of = date(2025, 1, 6)
    rows = [
        {"account_status": "confirmed", "expected_cash_date": None, "within_horizon": True, "amount_paise": 1000},
        {"account_status": "confirmed", "expected_cash_date": None, "within_horizon": True, "amount_paise": 500},
        {"account_status": "projected", "expected_cash_date": "2025-01-08", "within_horizon": True, "amount_paise": 300},
    ]
    chart_data = api_client.build_forecast_chart_data(rows, as_of)
    assert chart_data["2025-01-06"] == {"confirmed": 1500, "projected": 0}
    assert chart_data["2025-01-08"] == {"confirmed": 0, "projected": 300}
    assert list(chart_data.keys()) == sorted(chart_data.keys())


# ---------------------------------------------------------------------------
# Test 12 -- chart data-shaping drops rows outside the forecast horizon
# (AC6)
# ---------------------------------------------------------------------------

def test_build_forecast_chart_data_filters_out_rows_outside_horizon():
    as_of = date(2025, 1, 6)
    rows = [
        {"account_status": "projected", "expected_cash_date": "2025-02-01", "within_horizon": False, "amount_paise": 999},
    ]
    assert api_client.build_forecast_chart_data(rows, as_of) == {}


# ---------------------------------------------------------------------------
# Test 13 -- two runs triggered through api_client stay independently
# queryable, the data layer half of side-by-side viewing (AC7)
# ---------------------------------------------------------------------------

def test_api_client_two_triggered_runs_are_independently_queryable(dashboard_client):
    id1 = api_client.trigger_batch_run("frozen")["batch_run_id"]
    id2 = api_client.trigger_batch_run("frozen")["batch_run_id"]
    assert id1 != id2
    for batch_run_id in (id1, id2):
        status = api_client.get_status(batch_run_id)
        assert status["total"] == FROZEN_TOTAL_GROUND_TRUTH_ENTRIES


# ---------------------------------------------------------------------------
# Test 14 -- initial render: no crash, sidebar trigger controls present, and
# an info message tells the user there's nothing to view yet
# ---------------------------------------------------------------------------

def test_dashboard_initial_render_shows_trigger_controls(dashboard_client):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()

    assert not at.exception
    assert at.selectbox(key="trigger_source")
    assert at.button(key="trigger_button")
    assert at.text_input(key="manual_batch_run_id")
    assert any("Trigger a batch run" in i.value for i in at.info)


# ---------------------------------------------------------------------------
# Test 15 -- clicking "Trigger run" for the frozen source shows the correct
# match-rate summary metrics (AC1, AC3)
# ---------------------------------------------------------------------------

def test_dashboard_trigger_frozen_run_shows_match_rate_summary(dashboard_client):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button(key="trigger_button").click()
    at.run()

    assert not at.exception
    metric_values = {m.label: m.value for m in at.metric}
    assert metric_values.get("Fast path") == str(FROZEN_FAST_PATH_COUNT)
    assert metric_values.get("Agent resolved") == str(FROZEN_AGENT_RESOLVED_COUNT)
    assert metric_values.get("Honest exception") == str(FROZEN_HONEST_EXCEPTION_COUNT)
    assert metric_values.get("Total") == str(FROZEN_TOTAL_GROUND_TRUTH_ENTRIES)


# ---------------------------------------------------------------------------
# Test 16 -- the manual batch_run_id box pulls an already-triggered run
# (created outside the UI) into view without re-triggering it (AC2)
# ---------------------------------------------------------------------------

def test_dashboard_manual_batch_run_id_pulls_existing_run_into_view(dashboard_client):
    from streamlit.testing.v1 import AppTest

    existing_id = api_client.trigger_batch_run("frozen")["batch_run_id"]

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.text_input(key="manual_batch_run_id").set_value(existing_id)
    at.button(key="add_manual_button").click()
    at.run()

    assert not at.exception
    assert any("Added batch run" in s.value for s in at.success)
    assert existing_id in at.session_state["runs"]
    assert existing_id in at.multiselect(key="run_selector").value


# ---------------------------------------------------------------------------
# Test 17 -- an unknown manual batch_run_id shows a clean error and does not
# crash the page (AC2, AC8)
# ---------------------------------------------------------------------------

def test_dashboard_manual_batch_run_id_unknown_shows_error_not_crash(dashboard_client):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.text_input(key="manual_batch_run_id").set_value(str(uuid.uuid4()))
    at.button(key="add_manual_button").click()
    at.run()

    assert not at.exception
    assert any("Unknown batch_run_id" in e.value for e in at.error)


# ---------------------------------------------------------------------------
# Test 18 -- the honest exception list and the full trial balance table both
# render for a triggered run, with no rows silently swallowed (AC4, AC5)
# ---------------------------------------------------------------------------

def test_dashboard_exception_and_trial_balance_tables_render(dashboard_client):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button(key="trigger_button").click()
    at.run()

    assert not at.exception

    # st.dataframe only registers a lookup-able key when on_select is
    # active (confirmed by reading streamlit's elements/arrow.py -- key is
    # silently a no-op on proto.id otherwise), so with exactly one run
    # selected the two dataframes are identified by their fixed render
    # order instead: exceptions, then trial balance (see app.py).
    dataframes = list(at.dataframe)
    assert len(dataframes) == 2
    exceptions_table, trial_balance_table = dataframes[0].value, dataframes[1].value

    assert len(exceptions_table) == FROZEN_HONEST_EXCEPTION_COUNT
    assert "category" in exceptions_table.columns
    assert "TOTAL" in trial_balance_table["account_code"].values


# ---------------------------------------------------------------------------
# Test 19 -- the forecast chart section executes without raising for a
# triggered run (AC6). AppTest cannot inspect st.bar_chart's rendered
# series (confirmed: streamlit.testing.v1's AppTest exposes no chart-data
# accessor) -- a human must still glance at the real chart once; this test
# is the honest limit of what the harness can prove.
# ---------------------------------------------------------------------------

def test_dashboard_forecast_chart_section_renders_without_exception(dashboard_client):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button(key="trigger_button").click()
    at.run()

    assert not at.exception


# ---------------------------------------------------------------------------
# Test 20 -- two runs selected in the run selector render independently,
# side by side, each keyed by its own batch_run_id (AC7)
# ---------------------------------------------------------------------------

def test_dashboard_two_runs_render_independently_side_by_side(dashboard_client):
    from streamlit.testing.v1 import AppTest

    id1 = api_client.trigger_batch_run("frozen")["batch_run_id"]
    id2 = api_client.trigger_batch_run("frozen")["batch_run_id"]

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.text_input(key="manual_batch_run_id").set_value(id1)
    at.button(key="add_manual_button").click()
    at.run()
    at.text_input(key="manual_batch_run_id").set_value(id2)
    at.button(key="add_manual_button").click()
    at.run()
    at.multiselect(key="run_selector").set_value([id1, id2])
    at.run()

    assert not at.exception

    # Both runs are the same frozen dataset triggered twice, so their table
    # contents are identical -- what this test proves is that BOTH runs got
    # a full, independent render (nothing deduplicated or dropped), not that
    # their numbers differ. Two exception tables (17 rows, "category"
    # column) and two trial-balance tables ("TOTAL" row) must be present.
    dataframes = [d.value for d in at.dataframe]
    assert len(dataframes) == 4
    exception_shaped = [df for df in dataframes if "category" in df.columns]
    trial_balance_shaped = [df for df in dataframes if "account_code" in df.columns]
    assert len(exception_shaped) == 2
    assert all(len(df) == FROZEN_HONEST_EXCEPTION_COUNT for df in exception_shaped)
    assert len(trial_balance_shaped) == 2
    assert all("TOTAL" in df["account_code"].values for df in trial_balance_shaped)

    metric_values = [m.value for m in at.metric]
    assert metric_values.count(str(FROZEN_FAST_PATH_COUNT)) == 2
    assert metric_values.count(str(FROZEN_TOTAL_GROUND_TRUTH_ENTRIES)) == 2

    subheaders = {s.value for s in at.subheader}
    assert any(id1[:8] in s for s in subheaders)
    assert any(id2[:8] in s for s in subheaders)


# ---------------------------------------------------------------------------
# Test 21 -- the sidebar's settlement-cutoff checkbox actually drives a real,
# non-fabricated "projected" forecast row through the real trigger path
# (Layer 6 addendum: as_of-gated Stage 2 posting, approved after Layer 7 --
# see src/orchestration/batch_runner.py's module docstring). Before that
# fix, "projected" was unreachable via any real triggered run; this is the
# end-to-end proof that the dashboard's demo control actually achieves it.
# ---------------------------------------------------------------------------

def test_dashboard_trigger_with_cutoff_checkbox_produces_real_projected_forecast(dashboard_client):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.checkbox(key="trigger_gate_stage_2").set_value(True)
    at.run()
    at.date_input(key="trigger_as_of").set_value(date(2025, 1, 20))
    at.button(key="trigger_button").click()
    at.run()

    assert not at.exception
    batch_run_id = list(at.session_state["runs"].keys())[0]

    metric_values = {m.label: m.value for m in at.metric}
    resolved_total = int(metric_values["Fast path"]) + int(metric_values["Agent resolved"])
    assert resolved_total < FROZEN_FAST_PATH_COUNT + FROZEN_AGENT_RESOLVED_COUNT, (
        "the cutoff must actually have gated some settlements this run"
    )

    # Confirm through the real API (same as project_cashflow would compute)
    # that this run's ledger genuinely carries an in-flight order.
    forecast_rows = api_client.get_forecast(batch_run_id, as_of=date(2025, 1, 20), horizon_days=7)
    statuses = {row["account_status"] for row in forecast_rows}
    assert "projected" in statuses
    assert "confirmed" in statuses
