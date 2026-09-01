"""Layer 7: Streamlit dashboard.

Calls src/api/main.py (Layer 6) over HTTP via src/dashboard/api_client.py --
this module never imports src.ledger, src.matching, or src.forecast
directly (docs/plan.md's modular-monolith transport boundary: the dashboard
reaches ledger/matching/forecast state only through the API layer, same as
any other HTTP client would).

Views: a sidebar trigger form (frozen or live seed), a manual batch_run_id
entry box to pull an already-triggered run into view without re-triggering
it, a run selector, and -- per selected run, side by side in st.columns --
a match-rate summary, the honest exception list (category + full note), the
full trial balance table, and a confirmed-vs-projected cash forecast chart.
batch_run_id scoping (Layer 3) is what makes the frozen run and a live seed
run viewable side by side without either destroying the other.
"""
from __future__ import annotations

from datetime import date

import polars as pl
import streamlit as st

from src.dashboard import api_client

st.set_page_config(page_title="AI Finance Controller", layout="wide")
st.title("AI Finance Controller -- Reconciliation Dashboard")

if "runs" not in st.session_state:
    st.session_state.runs = {}  # batch_run_id -> display label

with st.sidebar:
    st.header("Trigger a batch run")
    source = st.selectbox("Source", ["frozen", "seed"], key="trigger_source")
    seed_value = None
    records_value = 100
    if source == "seed":
        seed_value = st.number_input("Seed", min_value=0, value=42, step=1, key="trigger_seed")
        records_value = st.number_input("Records", min_value=1, value=100, step=1, key="trigger_records")
    if st.button("Trigger run", key="trigger_button"):
        try:
            summary = api_client.trigger_batch_run(
                source,
                seed=int(seed_value) if seed_value is not None else None,
                records=int(records_value),
            )
            new_id = summary["batch_run_id"]
            label = f"seed={int(seed_value)}" if source == "seed" else "frozen"
            st.session_state.runs[new_id] = f"{label} -- {new_id[:8]}"
            st.success(f"Triggered batch run {new_id}")
        except api_client.ApiClientError as exc:
            st.error(f"Failed to trigger run: {exc.detail}")

    st.header("Add an existing run")
    manual_id = st.text_input("batch_run_id", key="manual_batch_run_id")
    if st.button("Add to view", key="add_manual_button"):
        if not manual_id:
            st.error("Enter a batch_run_id first.")
        else:
            try:
                api_client.get_status(manual_id)  # validates the run exists
                st.session_state.runs.setdefault(manual_id, f"manual -- {manual_id[:8]}")
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
            status = api_client.get_status(batch_run_id)
        except api_client.ApiClientError as exc:
            st.error(f"Could not load status: {exc.detail}")
            continue

        st.markdown("**Match rate summary**")
        metric_cols = st.columns(4)
        metric_cols[0].metric("Fast path", status["fast_path"])
        metric_cols[1].metric("Agent resolved", status["agent_resolved"])
        metric_cols[2].metric("Honest exception", status["honest_exception"])
        metric_cols[3].metric("Total", status["total"])

        st.markdown("**Honest exceptions**")
        exceptions = api_client.get_exceptions(batch_run_id)
        if exceptions:
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
        else:
            st.write("No honest exceptions.")

        st.markdown("**Trial balance**")
        trial_balance_rows = api_client.get_trial_balance(batch_run_id)
        st.dataframe(pl.DataFrame(trial_balance_rows))

        st.markdown("**Cash forecast (confirmed vs. projected)**")
        forecast_rows = api_client.get_forecast(batch_run_id, as_of=as_of, horizon_days=7)
        chart_data = api_client.build_forecast_chart_data(forecast_rows, as_of)
        if chart_data:
            chart_df = pl.DataFrame(
                [
                    {"date": d, "confirmed": v["confirmed"], "projected": v["projected"]}
                    for d, v in chart_data.items()
                ]
            )
            st.bar_chart(chart_df, x="date", y=["confirmed", "projected"])
        else:
            st.write("No in-flight cash to project.")
