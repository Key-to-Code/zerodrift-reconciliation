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

import re
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
# Test 2b -- trigger_batch_run's POST explicitly overrides the shared
# client's 30s default timeout. Real bug: a live seed trigger runs the real
# agent synchronously for every non-fast-path record before the response
# returns, which can genuinely take several minutes -- the shared client's
# general 30s timeout (correct for every other, always-fast DB-read
# endpoint) was silently too short for this one. A stub client/response,
# never a real network call, so this proves the override is actually passed
# through without waiting out either timeout.
# ---------------------------------------------------------------------------

def test_trigger_batch_run_passes_a_longer_timeout_than_the_client_default():
    captured = {}

    class _StubResponse:
        status_code = 200

        def json(self):
            return {"batch_run_id": "stub"}

    class _StubClient:
        def post(self, url, json=None, timeout=None):
            captured["timeout"] = timeout
            return _StubResponse()

    original_factory = api_client._client_factory
    api_client.set_client_factory(lambda: _StubClient())
    try:
        api_client.trigger_batch_run("frozen")
    finally:
        api_client.set_client_factory(original_factory)

    assert captured["timeout"] == api_client.TRIGGER_BATCH_RUN_TIMEOUT_SECONDS
    assert captured["timeout"] > 30.0, "must genuinely exceed the shared client's default timeout"


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
# Test 15 -- clicking "Trigger run" for the frozen source, then advancing
# through the gate via "View Overview ->", shows the correct match-rate
# summary metrics (AC1, AC3). REWRITTEN for true gated navigation (mockup-
# replication pass, explicit user sign-off) -- metrics no longer render on
# the same script pass as the trigger; reaching them now requires the same
# navigation click a real user would make.
# ---------------------------------------------------------------------------

def test_dashboard_trigger_frozen_run_shows_match_rate_summary(dashboard_client):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button(key="trigger_button").click()
    at.run()
    at.button(key="goto_overview_button").click()
    at.run()

    assert not at.exception
    metric_values = {m.label: m.value for m in at.metric}
    assert metric_values.get("Fast path") == str(FROZEN_FAST_PATH_COUNT)
    assert metric_values.get("Agent resolved") == str(FROZEN_AGENT_RESOLVED_COUNT)
    assert metric_values.get("Honest exception") == str(FROZEN_HONEST_EXCEPTION_COUNT)
    assert metric_values.get("Total") == str(FROZEN_TOTAL_GROUND_TRUTH_ENTRIES)


# ---------------------------------------------------------------------------
# Test 16 -- the manual batch_run_id box pulls an already-triggered run
# (created outside the UI) into view without re-triggering it (AC2).
# TOUCHED beyond the 5 explicitly-approved tests (15/18/19/20/21) for the
# same root cause as those: `run_selector` only exists once the page has
# left the Run view (mockup-replication pass, true gated navigation) --
# added one nav click, changed nothing else about what this test proves.
# ---------------------------------------------------------------------------

def test_dashboard_manual_batch_run_id_pulls_existing_run_into_view(dashboard_client):
    from streamlit.testing.v1 import AppTest

    existing_id = api_client.trigger_batch_run("frozen")["batch_run_id"]

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.text_input(key="manual_batch_run_id").set_value(existing_id)
    at.button(key="add_manual_button").click()
    at.run()

    # st.success() is transient -- only rendered on the script run where the
    # click itself happened -- so it must be checked before navigating away,
    # not after (navigating away is a separate run, on which
    # add_manual_button is no longer "just clicked").
    assert not at.exception
    assert any("Added batch run" in s.value for s in at.success)
    assert existing_id in at.session_state["runs"]

    at.button(key="nav_overview").click()
    at.run()
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
# render for a triggered run, with no rows silently swallowed (AC4, AC5).
# REWRITTEN for true gated navigation (mockup-replication pass, explicit
# user sign-off): Exceptions and Ledger are now separate gated views, so
# they're checked one navigation at a time rather than on one shared page.
# The old assertions read them off `at.dataframe` (a plain summary table);
# that summary table was replaced by real per-row st.expander widgets (real
# click-to-expand interactivity, which raw HTML can't do without JS) plus a
# custom HTML ledger table (needed for the TOTAL row's checkmark badge, see
# theme.py's .ledger-total-row/.balanced-badge) -- so the new assertions
# read `at.expander`/category-pill markdown and the ledger HTML string
# instead, but prove the identical claim: no exception or ledger row is
# silently swallowed.
# ---------------------------------------------------------------------------

def test_dashboard_exception_and_trial_balance_tables_render(dashboard_client):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button(key="trigger_button").click()
    at.run()

    at.button(key="nav_exceptions").click()
    at.run()
    assert not at.exception
    assert len(at.expander) == FROZEN_HONEST_EXCEPTION_COUNT
    pill_markdowns = [m.value for m in at.markdown if '<span class="category-pill">' in m.value]
    assert len(pill_markdowns) == FROZEN_HONEST_EXCEPTION_COUNT

    at.button(key="nav_ledger").click()
    at.run()
    assert not at.exception
    ledger_markdowns = [m.value for m in at.markdown if 'class="ledger-table"' in m.value]
    assert len(ledger_markdowns) == 1
    assert "TOTAL" in ledger_markdowns[0]


# ---------------------------------------------------------------------------
# Test 19 -- the forecast chart section executes without raising for a
# triggered run (AC6). AppTest cannot inspect an Altair chart's rendered
# series data any more than it could st.bar_chart's -- a human must still
# glance at the real chart once; this test is the honest limit of what the
# harness can prove. REWRITTEN for true gated navigation (mockup-replication
# pass, explicit user sign-off): the forecast section only renders after
# navigating to the Forecast view.
# ---------------------------------------------------------------------------

def test_dashboard_forecast_chart_section_renders_without_exception(dashboard_client):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button(key="trigger_button").click()
    at.run()
    at.button(key="nav_forecast").click()
    at.run()

    assert not at.exception


# ---------------------------------------------------------------------------
# Test 20 -- two runs selected in the run selector render independently,
# side by side, each keyed by its own batch_run_id (AC7). REWRITTEN for
# true gated navigation (mockup-replication pass, explicit user sign-off):
# nav governs WHICH section shows; multi-run side-by-side stays governed by
# how many runs are selected (compare-mode resolution) and is checked once
# per view, since only one view's content exists in a given script pass.
# The `run_selector` multiselect only exists once the page has left the
# Run view (any other view), so a nav click precedes the first use of it.
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

    at.button(key="nav_overview").click()
    at.run()
    at.multiselect(key="run_selector").set_value([id1, id2])
    at.run()

    assert not at.exception
    # Both runs are the same frozen dataset triggered twice, so their
    # numbers are identical -- what this proves is that BOTH runs got a
    # full, independent render (nothing deduplicated or dropped), not that
    # their numbers differ.
    metric_values = [m.value for m in at.metric]
    assert metric_values.count(str(FROZEN_FAST_PATH_COUNT)) == 2
    assert metric_values.count(str(FROZEN_TOTAL_GROUND_TRUTH_ENTRIES)) == 2
    subheaders = {s.value for s in at.subheader}
    assert any(id1[:8] in s for s in subheaders)
    assert any(id2[:8] in s for s in subheaders)

    at.button(key="nav_exceptions").click()
    at.run()
    assert not at.exception
    assert len(at.expander) == 2 * FROZEN_HONEST_EXCEPTION_COUNT
    pill_markdowns = [m.value for m in at.markdown if '<span class="category-pill">' in m.value]
    assert len(pill_markdowns) == 2 * FROZEN_HONEST_EXCEPTION_COUNT

    at.button(key="nav_ledger").click()
    at.run()
    assert not at.exception
    ledger_markdowns = [m.value for m in at.markdown if 'class="ledger-table"' in m.value]
    assert len(ledger_markdowns) == 2
    assert all("TOTAL" in lm for lm in ledger_markdowns)


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
    at.button(key="nav_overview").click()
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


# ---------------------------------------------------------------------------
# Tests 22+ -- the design.md UI pass (Layer 7 follow-up). Only the subset
# that is independent of the sidebar nav / single-run-vs-compare-mode
# architecture question is implemented here -- that question was raised as a
# genuine blocker during implementation (see the layer report) and needs a
# decision before AC10-17/24-29/32-33/37-38 (nav structure, Run/Overview/
# Exceptions/Ledger/Forecast screen restructuring, Compare mode) can be
# built and tested honestly. Numbering below intentionally skips the slots
# that belong to that deferred work, rather than filling them with tests
# that don't prove anything real yet.
# ---------------------------------------------------------------------------

def _ui_facing_source_text() -> str:
    # Plain file reads, not `import` -- app.py is a Streamlit script with
    # top-level st.* calls (st.set_page_config, st.columns, ...) that raise
    # outside a real ScriptRunContext, so importing it directly (rather than
    # running it via AppTest) crashes instead of scanning it.
    dashboard_dir = Path(APP_PATH).resolve().parent
    paths = [dashboard_dir / "app.py", dashboard_dir / "theme.py", dashboard_dir / "tokens.py"]
    return "\n".join(p.read_text(encoding="utf-8") for p in paths)


# ---------------------------------------------------------------------------
# Test 22 -- no em dash character anywhere in the dashboard's own UI-facing
# source (app.py/theme.py/tokens.py) (AC8)
# ---------------------------------------------------------------------------

def test_no_em_dash_in_dashboard_ui_source():
    assert "—" not in _ui_facing_source_text()


# ---------------------------------------------------------------------------
# Test 23 -- the run label shown in the sidebar (trigger success, and the
# manually-added-run path) is a professional "Label · shortid" shape, not
# the old raw "manual -- a1b2c3d4"/"frozen -- a1b2c3d4" format (AC9)
# ---------------------------------------------------------------------------

def test_dashboard_run_labels_use_professional_format(dashboard_client):
    from streamlit.testing.v1 import AppTest

    professional_label = re.compile(r"^[A-Za-z][\w ]*· [0-9a-f]{8}$")

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button(key="trigger_button").click()
    at.run()
    triggered_id = list(at.session_state["runs"].keys())[0]
    triggered_label = at.session_state["runs"][triggered_id]
    assert professional_label.match(triggered_label), triggered_label
    assert " -- " not in triggered_label

    existing_id = api_client.trigger_batch_run("frozen")["batch_run_id"]
    at.text_input(key="manual_batch_run_id").set_value(existing_id)
    at.button(key="add_manual_button").click()
    at.run()
    manual_label = at.session_state["runs"][existing_id]
    assert professional_label.match(manual_label), manual_label
    assert " -- " not in manual_label


# ---------------------------------------------------------------------------
# Test 30 -- _build_unmatched_bank_line_note threads a real, non-fabricated
# candidate_order_id straight from DiscrepancyRecord.candidate_orders (the
# same list the agent is shown as a force-match temptation -- see
# discrepancy.py's find_candidate_orders) into the note, and
# api_client.parse_confidence_note_candidate reads it back out correctly.
# No DB, no live model call -- a hand-built record and a stub resolution
# (AC15's data-availability half, AC22)
# ---------------------------------------------------------------------------

def test_unmatched_bank_line_note_carries_real_candidate_order_id():
    from src.agent.discrepancy import BankCredit, CandidateOrder, DiscrepancyRecord
    from src.agent.resolution import AgentResolution
    from src.orchestration.batch_runner import _build_unmatched_bank_line_note

    record = DiscrepancyRecord(
        discrepancy_reason="unmatched_bank_line",
        bank_credits=[
            BankCredit(utr="UTR999", credited_amount_paise=50000, value_date="2025-01-10", narration="NEFT")
        ],
        candidate_orders=[
            CandidateOrder(
                order_id="ORD1045",
                settlement_net_amount_paise=50120,
                settlement_date="2025-01-09",
                payment_method="upi",
            )
        ],
    )
    resolution = AgentResolution(
        root_cause_code="UNRESOLVED",
        quantified_delta_paise=0,
        confidence_note="Amount/date close to ORD1045 but no narration evidence tying them together.",
    )

    note = _build_unmatched_bank_line_note(record, resolution)

    assert "candidate_order_id=ORD1045" in note
    assert api_client.parse_confidence_note_candidate(note) == "ORD1045"


# ---------------------------------------------------------------------------
# Test 30b -- 2+ surfaced candidates are comma-joined, not silently
# truncated to one (AC22)
# ---------------------------------------------------------------------------

def test_unmatched_bank_line_note_joins_multiple_candidates():
    from src.agent.discrepancy import BankCredit, CandidateOrder, DiscrepancyRecord
    from src.agent.resolution import AgentResolution
    from src.orchestration.batch_runner import _build_unmatched_bank_line_note

    record = DiscrepancyRecord(
        discrepancy_reason="unmatched_bank_line",
        bank_credits=[
            BankCredit(utr="UTR998", credited_amount_paise=50000, value_date="2025-01-10", narration="NEFT")
        ],
        candidate_orders=[
            CandidateOrder(order_id="ORD1045", settlement_net_amount_paise=50120, settlement_date="2025-01-09", payment_method="upi"),
            CandidateOrder(order_id="ORD1046", settlement_net_amount_paise=49980, settlement_date="2025-01-11", payment_method="card"),
        ],
    )
    resolution = AgentResolution(root_cause_code="UNRESOLVED", quantified_delta_paise=0, confidence_note="ambiguous")

    note = _build_unmatched_bank_line_note(record, resolution)
    assert api_client.parse_confidence_note_candidate(note) == "ORD1045,ORD1046"


# ---------------------------------------------------------------------------
# Test 31a -- parse_confidence_note_candidate returns "not applicable" for a
# note that predates this field, using the two LITERAL legacy formats
# src/orchestration/batch_runner.py actually produces: the settlement-
# discrepancy note (4-field, still current -- never carries a candidate
# concept) and the unmatched_bank_line note's shape BEFORE this change
# (3-field, no candidate_order_id at all). This is a different code path
# from test 31b's "field present but blank" case -- see the parser: an
# absent field falls through the loop to the trailing return, a blank-value
# field returns via the `or` fallback inside the loop. Both must be proven
# separately (AC22)
# ---------------------------------------------------------------------------

def test_parse_confidence_note_candidate_handles_true_legacy_notes():
    # batch_runner.py:253-254's format (build_settlement_discrepancy_queue) --
    # unchanged by this session's work, never had a candidate concept.
    settlement_discrepancy_format = (
        "discrepancy_reason=short_settlement; root_cause=UNRESOLVED; delta_paise=0; stub note text"
    )
    assert api_client.parse_confidence_note_candidate(settlement_discrepancy_format) == "not applicable"

    # batch_runner.py:310-312's format as it existed BEFORE this session's
    # change to _build_unmatched_bank_line_note (no candidate_order_id field
    # at all) -- a genuinely legacy shape, not a new-format note with an
    # empty value.
    pre_change_unmatched_bank_line_format = (
        "discrepancy_reason=orphan; root_cause=UNRESOLVED; stub note text"
    )
    assert api_client.parse_confidence_note_candidate(pre_change_unmatched_bank_line_format) == "not applicable"


# ---------------------------------------------------------------------------
# Test 31b -- a NEW-format note where candidate_order_id is present but
# blank (no candidate was close enough in amount/date to be surfaced at
# all) also returns "not applicable", via the in-loop fallback rather than
# the true-legacy path above (AC22)
# ---------------------------------------------------------------------------

def test_parse_confidence_note_candidate_handles_blank_field():
    note = "discrepancy_reason=orphan; root_cause=UNRESOLVED; candidate_order_id=; no nearby candidates"
    assert api_client.parse_confidence_note_candidate(note) == "not applicable"


# ---------------------------------------------------------------------------
# Test 31c -- malformed/missing input doesn't raise (AC22)
# ---------------------------------------------------------------------------

def test_parse_confidence_note_candidate_handles_malformed_note():
    assert api_client.parse_confidence_note_candidate(None) == "not applicable"
    assert api_client.parse_confidence_note_candidate("") == "not applicable"
    assert api_client.parse_confidence_note_candidate("discrepancy_reason=orphan") == "not applicable"


# ---------------------------------------------------------------------------
# Test 34 -- per-run data (status/exceptions/trial balance/forecast) is
# fetched once per (batch_run_id[, as_of]) and served from st.cache_data on
# a later rerun that changes nothing about that run, not re-fetched over
# HTTP every time (AC18)
# ---------------------------------------------------------------------------

def test_dashboard_caches_status_across_unrelated_reruns(dashboard_client):
    """REWRITTEN for true gated navigation (mockup-replication pass,
    explicit user sign-off): _load_status is only ever called on a
    non-Run view (it's the per-run loop's first fetch, inside the `else`
    branch), so a nav click is now required before it fires at all."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.cache_data.clear()
    call_count = {"n": 0}
    original_get_status = api_client.get_status

    def counting_get_status(batch_run_id):
        call_count["n"] += 1
        return original_get_status(batch_run_id)

    api_client.get_status = counting_get_status
    try:
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.button(key="trigger_button").click()
        at.run()
        at.button(key="nav_overview").click()
        at.run()
        calls_after_nav = call_count["n"]
        assert calls_after_nav >= 1

        # A rerun that touches an unrelated widget (still on Overview, same
        # batch_run_id) must not re-fetch this same run's status over HTTP
        # again.
        at.run()
        assert call_count["n"] == calls_after_nav, (
            "get_status was called again on a rerun with the same batch_run_id -- "
            "st.cache_data should have served it from cache"
        )
    finally:
        api_client.get_status = original_get_status
        st.cache_data.clear()


# ---------------------------------------------------------------------------
# Test 35 -- every cached data-loading wrapper in app.py shows a spinner on
# a cache miss (AC19). Implemented via st.cache_data's own show_spinner
# parameter rather than a separate st.spinner() context -- a cleaner,
# equally-compliant idiom for exactly this "network call behind a cache"
# shape; noted here since the originally-discussed test wording assumed a
# bare st.spinner() call.
# ---------------------------------------------------------------------------

def test_app_cached_loaders_show_spinner_on_cache_miss():
    source = Path(APP_PATH).read_text(encoding="utf-8")
    for fn_name in ("_load_status", "_load_exceptions", "_load_trial_balance", "_load_forecast"):
        idx = source.index(f"def {fn_name}(")
        preceding = source[:idx]
        decorator_line = preceding.rstrip().splitlines()[-1]
        assert "st.cache_data" in decorator_line and "show_spinner=" in decorator_line, (
            f"{fn_name} must be an @st.cache_data(show_spinner=...) -decorated loader"
        )


# ---------------------------------------------------------------------------
# Test 36 -- st.rerun() is not called anywhere it isn't truly necessary
# (AC20). UPDATED from 0 to 1 exactly as this test's own prior comment
# anticipated: "expected to rise only when the deferred post-trigger
# navigation transition is actually built... update this test to assert
# the new, still-minimal count, not deleted." That transition is the
# mockup-replication pass's "View Overview ->" button -- its click sets
# session_state.view AFTER the script has already read that value into a
# local variable for this run, so without an immediate rerun the click
# would silently do nothing on screen until some unrelated later
# interaction. The sidebar nav buttons don't need this (their click
# happens earlier in the script than the `view` read), so this stays
# exactly one call, not one per navigation action.
# ---------------------------------------------------------------------------

def test_app_does_not_call_rerun_unnecessarily():
    source = Path(APP_PATH).read_text(encoding="utf-8")
    assert source.count("st.rerun(") == 1


# ---------------------------------------------------------------------------
# Blocker-resolution follow-up ("Hybrid: Run screen gated, rest on one
# page") -- tests for the newly-testable structural/behavioral pieces of
# AC10-17/21's visual half. Anything that needs a real browser to see
# (hover lift, card shadow, whether the stacked bar visually "looks
# professional") stays manually-verified, same honest limit as the existing
# bar_chart test -- these only assert on what AppTest can actually inspect:
# element presence/text/session_state, not rendered CSS.
# ---------------------------------------------------------------------------

def test_theme_defines_all_new_design_css_classes():
    theme_source = (Path(APP_PATH).resolve().parent / "theme.py").read_text(encoding="utf-8")
    for css_class in (
        "metric-card-positive", "metric-card-information", "metric-card-notice",
        "stacked-bar", "bar-segment-positive", "bar-segment-information", "bar-segment-notice",
        "category-pill", "ledger-total-row", "balanced-badge", "caution-banner", "centered-content",
        ":has(.bar-segment-positive:hover)",
    ):
        assert css_class in theme_source, css_class


def test_dashboard_run_section_shows_both_trigger_cards(dashboard_client):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    assert not at.exception
    subheaders = {s.value for s in at.subheader}
    assert "Run the frozen benchmark" in subheaders
    assert "Bring your own seed" in subheaders


def test_dashboard_sections_gate_to_exactly_one_view_at_a_time(dashboard_client):
    """RENAMED and REWRITTEN (was test_dashboard_overview_exceptions_ledger_
    forecast_sections_all_render) for true gated navigation (mockup-
    replication pass, explicit user sign-off): the previous one-page
    architecture made "all render together" the claim worth proving; under
    true gating that claim is now false by design, and the claim worth
    proving is its opposite -- navigating to a section shows THAT section's
    header and no other section's, i.e. gating actually gates."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button(key="trigger_button").click()
    at.run()

    section_headers = {
        "overview": "### Overview",
        "exceptions": "### Exceptions",
        "ledger": "### Ledger -- trial balance",
        "forecast": "### Forecast -- confirmed vs. projected",
    }
    for view_key, expected_header in section_headers.items():
        at.button(key=f"nav_{view_key}").click()
        at.run()
        assert not at.exception
        headers = {m.value for m in at.markdown}
        assert expected_header in headers, f"{view_key} view is missing its own header"
        for other_key, other_header in section_headers.items():
            if other_key != view_key:
                assert other_header not in headers, (
                    f"{other_header} leaked into the {view_key} view -- gating is broken"
                )


def test_dashboard_exceptions_show_category_pill(dashboard_client):
    """REWRITTEN for true gated navigation (mockup-replication pass,
    explicit user sign-off): the pills only render once the Exceptions
    view is active."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button(key="trigger_button").click()
    at.run()
    at.button(key="nav_exceptions").click()
    at.run()

    assert not at.exception
    # Filter on the actual rendered <span>, not just the substring
    # "category-pill" -- inject_theme()'s own injected <style> block is
    # also an at.markdown element and defines .category-pill, which would
    # otherwise inflate this count by one.
    pill_markdowns = [m.value for m in at.markdown if '<span class="category-pill">' in m.value]
    assert len(pill_markdowns) == FROZEN_HONEST_EXCEPTION_COUNT


def test_dashboard_frozen_run_surfaces_real_candidate_for_at_least_one_exception(dashboard_client):
    """End-to-end proof (not just the unit-level parser/note-builder tests)
    that a real triggered frozen run's adversarial_trap exceptions actually
    reach the Exceptions section's per-row detail with real candidate data --
    the generator deliberately jitters an adversarial_trap decoy within
    +/-200 paise/+/-1 day of a real twin order's settlement, strictly inside
    find_candidate_orders' wider +/-500 paise/+/-2 day tolerance, so this is
    a deterministic, not lucky, assertion for seed=42.

    REWRITTEN for true gated navigation (mockup-replication pass, explicit
    user sign-off): reaching the Exceptions view now needs a nav click.
    Also updated for the mockup-styled candidate callout, which moved from
    a plain st.caption() to a markdown div (.caution-banner) so it reads as
    a highlighted box rather than a caption line -- check at.markdown, not
    at.caption."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button(key="trigger_button").click()
    at.run()
    at.button(key="nav_exceptions").click()
    at.run()

    assert not at.exception
    markdowns = [m.value for m in at.markdown]
    assert any("Candidate considered and correctly rejected" in m for m in markdowns)


def test_dashboard_ledger_shows_balanced_badge(dashboard_client):
    """REWRITTEN for true gated navigation and the new custom-HTML ledger
    table (mockup-replication pass, explicit user sign-off): Ledger is now
    a separate gated view, and the TOTAL row's checkmark badge sits next to
    the net figure itself ("0.00") rather than a literal word "Balanced" --
    checking for the badge span plus the balanced net figure is the
    equivalent, honest assertion for the new markup."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button(key="trigger_button").click()
    at.run()
    at.button(key="nav_ledger").click()
    at.run()

    assert not at.exception
    badge_markdowns = [m.value for m in at.markdown if "balanced-badge" in m.value]
    assert any("0.00" in b for b in badge_markdowns)


def test_dashboard_cutoff_caption_omitted_when_no_cutoff_applied(dashboard_client):
    """REWRITTEN for true gated navigation (mockup-replication pass,
    explicit user sign-off): the Source/Cutoff captions live in the
    Overview section, which now needs a nav click to reach."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button(key="trigger_button").click()
    at.run()
    at.button(key="nav_overview").click()
    at.run()

    assert not at.exception
    captions = [c.value for c in at.caption]
    assert not any("Cutoff (as_of)" in c for c in captions)
    assert any(c.startswith("Source:") for c in captions)


def test_dashboard_cutoff_caption_shown_when_cutoff_applied(dashboard_client):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.checkbox(key="trigger_gate_stage_2").set_value(True)
    at.run()
    at.date_input(key="trigger_as_of").set_value(date(2025, 1, 20))
    at.button(key="trigger_button").click()
    at.run()
    at.button(key="nav_overview").click()
    at.run()

    assert not at.exception
    captions = [c.value for c in at.caption]
    assert any("Cutoff (as_of): 2025-01-20" in c for c in captions)


def test_dashboard_empty_manual_id_shows_caution_banner_not_st_error(dashboard_client):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.button(key="add_manual_button").click()
    at.run()

    assert not at.exception
    assert not list(at.error)
    banners = [m.value for m in at.markdown if "caution-banner" in m.value]
    assert any("Enter a batch_run_id first" in b for b in banners)


def test_dashboard_unknown_manual_id_still_uses_native_st_error(dashboard_client):
    """Regression guard for the ONE error path deliberately left on native
    st.error() rather than the new caution-banner -- test 17 asserts on
    at.error directly, so this path can't be restyled without breaking it."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(APP_PATH, default_timeout=30)
    at.run()
    at.text_input(key="manual_batch_run_id").set_value(str(uuid.uuid4()))
    at.button(key="add_manual_button").click()
    at.run()

    assert not at.exception
    assert any("Unknown batch_run_id" in e.value for e in at.error)
    banners = [m.value for m in at.markdown if "caution-banner" in m.value]
    assert not any("Unknown batch_run_id" in b for b in banners)


def test_dashboard_exceptions_empty_state_shows_calm_message(dashboard_client):
    """REWRITTEN for true gated navigation (mockup-replication pass,
    explicit user sign-off): needs a nav click to reach the Exceptions
    view before the empty-state message can render."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.cache_data.clear()
    original_get_exceptions = api_client.get_exceptions
    api_client.get_exceptions = lambda batch_run_id: []
    try:
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.button(key="trigger_button").click()
        at.run()
        at.button(key="nav_exceptions").click()
        at.run()

        assert not at.exception
        texts = [m.value for m in at.markdown]
        assert any("No honest exceptions on this run." in t for t in texts)
    finally:
        api_client.get_exceptions = original_get_exceptions
        st.cache_data.clear()


def test_dashboard_forecast_empty_state_shows_settled_note(dashboard_client):
    """REWRITTEN for true gated navigation (mockup-replication pass,
    explicit user sign-off): needs a nav click to reach the Forecast view
    before the empty-state message can render."""
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    st.cache_data.clear()
    original_get_forecast = api_client.get_forecast
    api_client.get_forecast = lambda batch_run_id, as_of, horizon_days=7: []
    try:
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        at.button(key="trigger_button").click()
        at.run()
        at.button(key="nav_forecast").click()
        at.run()

        assert not at.exception
        texts = [m.value for m in at.markdown]
        assert any("fully settled" in t for t in texts)
    finally:
        api_client.get_forecast = original_get_forecast
        st.cache_data.clear()
