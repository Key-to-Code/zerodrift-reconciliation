"""Layer 7: Streamlit dashboard.

Calls src/api/main.py (Layer 6) over HTTP via src/dashboard/api_client.py --
this module never imports src.ledger, src.matching, or src.forecast
directly (docs/plan.md's modular-monolith transport boundary: the dashboard
reaches ledger/matching/forecast state only through the API layer, same as
any other HTTP client would).

Layout (design.md, Layer 7 UI follow-up): a Run section at the top of the
main content area (two trigger cards, a collapsed cutoff expander, a manual
batch_run_id load box), then -- per selected run, side by side in
st.columns when 2+ runs are selected, one full-width column when exactly 1
is selected -- an Overview section (match-rate cards + stacked bar +
live-computed caption), an Exceptions section (table + per-row detail),
a Ledger section (trial balance), and a Forecast section (confirmed vs.
projected). This is deliberately NOT a click-gated multi-screen flow: every
section renders in the same script pass once a run exists, which is what
lets tests 15/18/19/20/21 (click trigger -> at.run() -> assert immediately)
keep passing unmodified -- see the Layer 7 follow-up conversation for why a
literal gated-navigation reading of design.md's Run screen was rejected.

Known, deliberately-scoped gaps (flagged, not silently shipped):
- The Exceptions table has no Amount column. ExceptionRecord (src/api/main.py)
  doesn't carry a settlement/bank-credit amount today -- adding one is a
  small, real backend field, same shape as the candidate_order_id addendum,
  but wasn't itself pre-approved, so it's omitted rather than fabricated.
- Forecast bars use the real brand palette instead of Streamlit's default
  purple, but do not carry a grayscale-safe hatch/opacity distinction --
  st.bar_chart has no per-series opacity/pattern hook; doing that honestly
  needs st.altair_chart, deferred as separate scope.
- Card-hover highlighting the matching Forecast chart series (not just the
  card itself) has the same st.bar_chart limitation.
"""
from __future__ import annotations

from datetime import date

import polars as pl
import streamlit as st

from src.common.money import from_paise
from src.dashboard import api_client
from src.dashboard.theme import inject_theme


@st.cache_data(show_spinner="Loading status...")
def _load_status(batch_run_id: str) -> dict:
    return api_client.get_status(batch_run_id)


@st.cache_data(show_spinner="Loading exceptions...")
def _load_exceptions(batch_run_id: str) -> list[dict]:
    return api_client.get_exceptions(batch_run_id)


@st.cache_data(show_spinner="Loading trial balance...")
def _load_trial_balance(batch_run_id: str) -> list[dict]:
    return api_client.get_trial_balance(batch_run_id)


@st.cache_data(show_spinner="Loading forecast...")
def _load_forecast(batch_run_id: str, as_of: date, horizon_days: int = 7) -> list[dict]:
    return api_client.get_forecast(batch_run_id, as_of=as_of, horizon_days=horizon_days)


def _caution_banner(message: str) -> None:
    """A calm caution-toned inline banner (design.md 4: "never a red error
    box") for error paths NOT covered by an existing at.error()-based test
    assertion. test_dashboard_manual_batch_run_id_unknown_shows_error_not_crash
    (test 17) asserts on at.error directly, so that ONE call site -- the
    "Unknown batch_run_id" path below -- deliberately stays a native
    st.error() instead of this banner; every other error path uses it."""
    st.markdown(f'<div class="caution-banner">{message}</div>', unsafe_allow_html=True)


st.set_page_config(page_title="ZeroDrift", layout="wide")
inject_theme()
st.title("ZeroDrift -- Reconciliation Dashboard")

if "runs" not in st.session_state:
    st.session_state.runs = {}  # batch_run_id -> display label
if "run_meta" not in st.session_state:
    st.session_state.run_meta = {}  # batch_run_id -> {"source": ..., "cutoff": date|None}

with st.sidebar:
    st.caption("ZeroDrift")
    st.markdown("**On this page**")
    st.markdown("- Run\n- Overview\n- Exceptions\n- Ledger\n- Forecast")

st.header("Run")
run_col_a, run_col_b = st.columns(2)
active_source = st.session_state.get("trigger_source", "frozen")

with run_col_a:
    with st.container(key="run-card-frozen", border=True):
        st.subheader("Run the frozen benchmark")
        st.caption(
            "100 synthetic records, seed 42, committed to this repo -- "
            "reproduce our numbers yourself."
        )
        if active_source != "frozen":
            st.caption('Switch Source to "frozen" below to use this card.')

with run_col_b:
    with st.container(key="run-card-seed", border=True):
        st.subheader("Bring your own seed")
        st.caption(
            "Generates a fresh, never-before-seen batch with the same "
            "category distribution. Full agent verification may take "
            "longer and calls a live model."
        )
        if active_source != "seed":
            st.caption('Switch Source to "seed" below to use this card.')

source = st.selectbox("Source", ["frozen", "seed"], key="trigger_source")
seed_value = None
records_value = 100
if source == "seed":
    seed_value = st.number_input("Seed", min_value=0, value=42, step=1, key="trigger_seed")
    records_value = st.number_input("Records", min_value=1, value=100, step=1, key="trigger_records")

with st.expander("Advanced: cutoff date", expanded=False):
    st.caption(
        "Limits ledger settlement to a specific date -- use this to see a "
        "genuine in-progress reconciliation state, e.g. for the forecast "
        "chart's confirmed vs. projected split."
    )
    gate_stage_2 = st.checkbox(
        "Limit settlement posting to a cutoff date",
        key="trigger_gate_stage_2",
    )
    trigger_as_of = None
    if gate_stage_2:
        trigger_as_of = st.date_input(
            "Settlement cutoff (as_of)", value=date(2025, 1, 20), key="trigger_as_of"
        )

trigger_label = "Trigger frozen batch" if source == "frozen" else "Trigger live batch"
if st.button(trigger_label, key="trigger_button"):
    with st.spinner("Calling agent, this may take a minute..." if source == "seed" else "Posting to ledger..."):
        try:
            summary = api_client.trigger_batch_run(
                source,
                seed=int(seed_value) if seed_value is not None else None,
                records=int(records_value),
                as_of=trigger_as_of,
            )
            new_id = summary["batch_run_id"]
            label = f"Seed {int(seed_value)}" if source == "seed" else "Frozen dataset"
            st.session_state.runs[new_id] = f"{label} · {new_id[:8]}"
            st.session_state.run_meta[new_id] = {"source": source, "cutoff": trigger_as_of}
            st.success(f"Triggered batch run {new_id}")
        except api_client.ApiClientError as exc:
            _caution_banner(f"Failed to trigger run: {exc.detail}")

st.markdown("**Or load an existing run**")
manual_id = st.text_input(
    "Run ID",
    key="manual_batch_run_id",
    placeholder="e.g. a1b2c3d4-5678-...",
    help="Paste a batch_run_id from a previous run to pull it into view without re-triggering it.",
)
if st.button("Add to view", key="add_manual_button"):
    if not manual_id:
        _caution_banner("Enter a batch_run_id first.")
    else:
        try:
            api_client.get_status(manual_id)  # validates the run exists
            st.session_state.runs.setdefault(manual_id, f"Existing run · {manual_id[:8]}")
            st.success(f"Added batch run {manual_id}")
        except api_client.ApiClientError as exc:
            st.error(f"Unknown batch_run_id: {exc.detail}")

if not st.session_state.runs:
    st.info("Trigger a batch run or add an existing batch_run_id to begin.")
    st.stop()

selected_runs = st.multiselect(
    "Runs to view",
    options=list(st.session_state.runs.keys()),
    default=list(st.session_state.runs.keys()),
    format_func=lambda rid: st.session_state.runs.get(rid, rid),
    key="run_selector",
)

as_of = st.date_input(
    "Forecast as-of date",
    value=date(2025, 1, 6),
    key="forecast_as_of",
)

if not selected_runs:
    st.info("Select at least one run to view.")
    st.stop()

columns = st.columns(len(selected_runs))
for column, batch_run_id in zip(columns, selected_runs):
    with column:
        st.subheader(st.session_state.runs.get(batch_run_id, batch_run_id))

        try:
            status = _load_status(batch_run_id)
        except api_client.ApiClientError as exc:
            _caution_banner(f"Could not load status: {exc.detail}")
            continue

        # -- Overview --------------------------------------------------
        with st.container(key=f"overview-section-{batch_run_id}"):
            st.markdown("### Overview")

            total = status["total"] or 1
            fast_pct = round(100 * status["fast_path"] / total)
            agent_pct = round(100 * status["agent_resolved"] / total)
            honest_pct = round(100 * status["honest_exception"] / total)

            card_cols = st.columns(3)
            with card_cols[0]:
                with st.container(key=f"metric-card-positive-{batch_run_id}"):
                    st.metric("Fast path", status["fast_path"], delta=f"{fast_pct}% of total", delta_color="off")
            with card_cols[1]:
                with st.container(key=f"metric-card-information-{batch_run_id}"):
                    st.metric("Agent resolved", status["agent_resolved"], delta=f"{agent_pct}% of total", delta_color="off")
            with card_cols[2]:
                with st.container(key=f"metric-card-notice-{batch_run_id}"):
                    st.metric("Honest exception", status["honest_exception"], delta=f"{honest_pct}% of total", delta_color="off")
            # Total kept as a plain metric outside the 3-card accent row --
            # it isn't one of the three semantic outcome colors.
            st.metric("Total", status["total"])

            st.markdown(
                f'<div class="stacked-bar">'
                f'<div class="bar-segment-positive" style="width:{fast_pct}%"></div>'
                f'<div class="bar-segment-information" style="width:{agent_pct}%"></div>'
                f'<div class="bar-segment-notice" style="width:{honest_pct}%"></div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                f"{status['fast_path']} resolved by deterministic matching alone -- no model call. "
                f"{status['agent_resolved']} required agent judgment. "
                f"{status['honest_exception']} could not be confidently resolved and were routed "
                f"to suspense rather than guessed."
            )

            # "Total records" is deliberately not repeated here as its own
            # st.metric -- it's already the "Total" card above, and a second
            # st.metric with the same value would silently double-count in
            # any at.metric-based assertion (caught by test 20 during this
            # implementation: it asserts the "100" value appears exactly
            # twice across 2 runs, not four times).
            meta = st.session_state.run_meta.get(batch_run_id, {})
            secondary_cols = st.columns(2 if meta.get("cutoff") else 1)
            secondary_cols[0].caption(f"Source: {meta.get('source', 'existing run').title()}")
            if meta.get("cutoff"):
                secondary_cols[1].caption(f"Cutoff (as_of): {meta['cutoff'].isoformat()}")

        # -- Exceptions --------------------------------------------------
        st.markdown("### Exceptions")
        exceptions = _load_exceptions(batch_run_id)
        if exceptions:
            st.caption(
                f"These {len(exceptions)} records could not be confidently matched or explained. "
                f"Each is preserved as a suspense entry in the ledger rather than a guess."
            )
            exceptions_df = pl.DataFrame(
                [
                    {
                        "order_id": row["order_id"],
                        "utr": row["utr"],
                        "category": api_client.parse_confidence_note_category(row["confidence_note"]),
                        "note": row["confidence_note"],
                    }
                    for row in exceptions
                ]
            )
            st.dataframe(exceptions_df)

            for row in exceptions:
                category = api_client.parse_confidence_note_category(row["confidence_note"])
                reason = api_client.parse_confidence_note_reason(row["confidence_note"])
                candidate = api_client.parse_confidence_note_candidate(row["confidence_note"])
                with st.expander(f"{row['order_id']} · {row['utr'] or 'no UTR'}"):
                    st.markdown(f'<span class="category-pill">{category}</span>', unsafe_allow_html=True)
                    st.write(reason or row["confidence_note"])
                    if candidate != "not applicable":
                        st.caption(f"Candidate considered and correctly rejected: {candidate}")
        else:
            st.write("No honest exceptions on this run.")

        # -- Ledger --------------------------------------------------
        st.markdown("### Ledger")
        trial_balance_rows = _load_trial_balance(batch_run_id)
        st.dataframe(pl.DataFrame(trial_balance_rows))
        total_row = next((r for r in trial_balance_rows if r["account_code"] == "TOTAL"), None)
        if total_row is not None and total_row["net_balance_paise"] == 0:
            st.markdown('<span class="balanced-badge">&#10003; Balanced -- net 0.00</span>', unsafe_allow_html=True)

        # -- Forecast --------------------------------------------------
        st.markdown('<div class="centered-content">', unsafe_allow_html=True)
        st.markdown("### Forecast")
        forecast_rows = _load_forecast(batch_run_id, as_of=as_of, horizon_days=7)
        chart_data = api_client.build_forecast_chart_data(forecast_rows, as_of)
        if chart_data:
            total_confirmed = from_paise(sum(v["confirmed"] for v in chart_data.values()))
            total_projected = from_paise(sum(v["projected"] for v in chart_data.values()))

            highlight_key = f"forecast_highlight_{batch_run_id}"
            if highlight_key not in st.session_state:
                st.session_state[highlight_key] = None

            summary_cols = st.columns(2)
            with summary_cols[0]:
                with st.container(key=f"forecast-card-confirmed-{batch_run_id}", border=True):
                    st.markdown(f'Confirmed: <span class="money-figure">₹{total_confirmed:,}</span>', unsafe_allow_html=True)
                    if st.button("Highlight confirmed", key=f"highlight_confirmed_{batch_run_id}"):
                        current = st.session_state[highlight_key]
                        st.session_state[highlight_key] = None if current == "confirmed" else "confirmed"
            with summary_cols[1]:
                with st.container(key=f"forecast-card-projected-{batch_run_id}", border=True):
                    st.markdown(f'Projected: <span class="money-figure">₹{total_projected:,}</span>', unsafe_allow_html=True)
                    if st.button("Highlight projected", key=f"highlight_projected_{batch_run_id}"):
                        current = st.session_state[highlight_key]
                        st.session_state[highlight_key] = None if current == "projected" else "projected"

            st.caption("Projected bars carry a +/-5% settlement-day slip, illustrative.")

            chart_df = pl.DataFrame(
                [
                    {"date": d, "confirmed": v["confirmed"], "projected": v["projected"]}
                    for d, v in chart_data.items()
                ]
            )
            st.bar_chart(
                chart_df,
                x="date",
                y=["confirmed", "projected"],
                color=["#00753B", "#0070A8"],  # tokens.COLOR_STATUS_POSITIVE / COLOR_STATUS_INFORMATION
                height=420,
            )
        else:
            st.write(
                "This run is fully settled -- nothing is currently projected. "
                "Trigger with a cutoff date to see an in-progress state."
            )
        st.markdown("</div>", unsafe_allow_html=True)
