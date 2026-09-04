"""Streamlit dashboard.

Calls src/api/main.py over HTTP via src/dashboard/api_client.py -- this
module never imports src.ledger, src.matching, or src.forecast directly
(modular-monolith transport boundary: ledger/matching/forecast state is
reached only through the API layer, same as any other HTTP client would).

Navigation is gated single-view: st.session_state.view is one of
"run"/"overview"/"exceptions"/"ledger"/"forecast", only the active view's
content renders, and the sidebar's nav buttons stay disabled until a run
is loaded. Compare mode (2+ runs selected) is not a separate nav state --
the active view still governs which section shows, rendered once per
selected run in side-by-side st.columns.

Known, deliberately-scoped gaps:
- The Exceptions table has no Amount column -- ExceptionRecord
  (src/api/main.py) doesn't carry a settlement/bank-credit amount today.
- The Forecast chart's projected bars are distinguished by color + reduced
  opacity + a dashed stroke outline, not a literal hatch fill -- Vega-Lite
  has no built-in repeating-pattern fill without a custom SVG pattern def.
"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

import altair as alt
import polars as pl
import streamlit as st

st.set_page_config(page_title="ZeroDrift Reconciliation", layout="wide")

from src.agent.rate_limiter import RECOMMENDED_MAX_LIVE_SEED_RECORDS
from src.common.money import format_inr, from_paise
from src.dashboard import api_client, tokens
from src.dashboard.theme import inject_theme

VIEWS = ("run", "overview", "exceptions", "ledger", "forecast")
VIEW_LABELS = {
    "run": "Run",
    "overview": "Overview",
    "exceptions": "Exceptions",
    "ledger": "Ledger",
    "forecast": "Forecast",
}


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
    """A calm caution-toned banner, never Streamlit's default red st.error --
    except the "Unknown batch_run_id" path below, which deliberately stays a
    native st.error() because a test asserts on it directly."""
    st.markdown(f'<div class="caution-banner">{message}</div>', unsafe_allow_html=True)


def _time_ago(moment: datetime) -> str:
    seconds = int((datetime.now() - moment).total_seconds())
    if seconds < 60:
        return "just now"
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} min ago"
    hours = minutes // 60
    return f"{hours}h ago"


def _run_badge(batch_run_id: str) -> None:
    meta = st.session_state.run_meta.get(batch_run_id, {})
    source = meta.get("source", "existing")
    if source == "frozen":
        source_label = "frozen dataset"
    elif source == "seed":
        source_label = f"seed {meta.get('seed')}" if meta.get("seed") is not None else "seed run"
    else:
        source_label = "existing run"
    when = meta.get("triggered_at") or meta.get("loaded_at")
    verb = "triggered" if meta.get("triggered_at") else "added"
    time_phrase = f" - {verb} {_time_ago(when)}" if when is not None else ""
    st.markdown(
        f'<div class="run-badge-bar">Viewing run '
        f'<span class="mono">{batch_run_id[:8]}...</span>'
        f' - {source_label}{time_phrase}</div>',
        unsafe_allow_html=True,
    )


def _render_overview(batch_run_id: str, status: dict, meta: dict) -> None:
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
            f"{status['fast_path']} resolved by deterministic matching alone - no model call. "
            f"{status['agent_resolved']} required agent judgment. "
            f"{status['honest_exception']} could not be confidently resolved and were routed "
            f"to suspense rather than guessed."
        )

        secondary_cols = st.columns(2 if meta.get("cutoff") else 1)
        secondary_cols[0].caption(f"Source: {meta.get('source', 'existing run').title()}")
        if meta.get("cutoff"):
            secondary_cols[1].caption(f"Cutoff (as_of): {meta['cutoff'].isoformat()}")


def _render_exceptions(batch_run_id: str) -> None:
    st.markdown("### Exceptions")
    exceptions = _load_exceptions(batch_run_id)
    if exceptions:
        st.caption(
            f"These {len(exceptions)} records could not be confidently matched or explained. "
            f"Each is preserved as a suspense entry in the ledger rather than a guess."
        )
        for row in exceptions:
            category = api_client.parse_confidence_note_category(row["confidence_note"])
            category_label = api_client.humanize_category(category)
            reason = api_client.parse_confidence_note_reason(row["confidence_note"])
            candidate = api_client.parse_confidence_note_candidate(row["confidence_note"])
            reference = row["utr"] or row["order_id"] or "unknown"
            with st.expander(f"{reference} - {category_label}"):
                st.markdown(f'<span class="category-pill">{category_label}</span>', unsafe_allow_html=True)
                st.write(reason or row["confidence_note"])
                if candidate != "not applicable":
                    st.markdown(
                        f'<div class="caution-banner">'
                        f'Candidate considered and correctly rejected: {candidate}</div>',
                        unsafe_allow_html=True,
                    )
    else:
        st.write("No honest exceptions on this run.")


def _render_ledger(batch_run_id: str) -> None:
    st.markdown("### Ledger - trial balance")
    st.caption(
        "Every account's closing debit/credit balance for this run. The TOTAL row "
        "must net to exactly zero for the books to balance."
    )
    trial_balance_rows = _load_trial_balance(batch_run_id)

    # One shared grid for the whole table (see theme.py's .ledger-table
    # comment) -- every cell is a direct child, never a per-row div, so
    # columns align across all rows.
    cells_html = ['<div class="ledger-table">']
    for i, label in enumerate(("Code", "Account", "Type", "Debit", "Credit", "Net")):
        right = " ledger-cell-right" if i >= 3 else ""
        cells_html.append(f'<div class="ledger-cell ledger-cell-header{right}">{label}</div>')

    total_row = None
    for row in trial_balance_rows:
        if row["account_code"] == "TOTAL":
            total_row = row
            continue
        cells_html.append(f'<div class="ledger-cell mono">{row["account_code"]}</div>')
        cells_html.append(f'<div class="ledger-cell">{row["account_name"]}</div>')
        cells_html.append(f'<div class="ledger-cell">{row["account_type"]}</div>')
        cells_html.append(
            f'<div class="ledger-cell mono ledger-cell-right">{format_inr(from_paise(row["debit_total_paise"]))}</div>'
        )
        cells_html.append(
            f'<div class="ledger-cell mono ledger-cell-right">{format_inr(from_paise(row["credit_total_paise"]))}</div>'
        )
        cells_html.append(
            f'<div class="ledger-cell mono ledger-cell-right">{format_inr(from_paise(row["net_balance_paise"]))}</div>'
        )
    if total_row is not None:
        net = from_paise(total_row["net_balance_paise"])
        badge = '<span class="balanced-badge">&#10003;</span>' if net == Decimal("0.00") else ""
        cells_html.append('<div class="ledger-cell ledger-total-row"></div>')
        cells_html.append('<div class="ledger-cell ledger-total-row">TOTAL</div>')
        cells_html.append('<div class="ledger-cell ledger-total-row"></div>')
        cells_html.append(
            f'<div class="ledger-cell ledger-total-row mono ledger-cell-right">'
            f'{format_inr(from_paise(total_row["debit_total_paise"]))}</div>'
        )
        cells_html.append(
            f'<div class="ledger-cell ledger-total-row mono ledger-cell-right">'
            f'{format_inr(from_paise(total_row["credit_total_paise"]))}</div>'
        )
        cells_html.append(
            f'<div class="ledger-cell ledger-total-row mono ledger-cell-right">{badge}{format_inr(net)}</div>'
        )
    cells_html.append('</div>')
    st.markdown("".join(cells_html), unsafe_allow_html=True)


_INDIAN_AXIS_LABEL_EXPR = (
    "datum.value == 0 ? '0' : "
    "abs(datum.value) >= 10000000 ? format(datum.value/10000000, '.2~f') + 'Cr' : "
    "abs(datum.value) >= 100000 ? format(datum.value/100000, '.2~f') + 'L' : "
    "abs(datum.value) >= 1000 ? format(datum.value/1000, '.1~f') + 'k' : "
    "format(datum.value, '.0f')"
)


def _render_forecast(batch_run_id: str, as_of: date) -> None:
    st.markdown('<div class="centered-content">', unsafe_allow_html=True)
    st.markdown("### Forecast - confirmed vs. projected")
    st.caption(
        "Cash already posted to CASH (confirmed) versus still in-flight through the "
        "settlement pipeline (projected), over the next 7 days."
    )
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
                st.markdown(
                    '<div class="forecast-card-label">Confirmed cash</div>'
                    f'<span class="money-figure">₹{format_inr(total_confirmed)}</span>',
                    unsafe_allow_html=True,
                )
                if st.button("Highlight confirmed", key=f"highlight_confirmed_{batch_run_id}"):
                    current = st.session_state[highlight_key]
                    st.session_state[highlight_key] = None if current == "confirmed" else "confirmed"
        with summary_cols[1]:
            with st.container(key=f"forecast-card-projected-{batch_run_id}", border=True):
                st.markdown(
                    '<div class="forecast-card-label">Projected cash</div>'
                    f'<span class="money-figure">₹{format_inr(total_projected)}</span>',
                    unsafe_allow_html=True,
                )
                if st.button("Highlight projected", key=f"highlight_projected_{batch_run_id}"):
                    current = st.session_state[highlight_key]
                    st.session_state[highlight_key] = None if current == "projected" else "projected"

        st.caption("Projected bars carry a +/-5% settlement-day slip, illustrative.")

        date_keys = list(chart_data.keys())
        date_labels = {d: datetime.strptime(d, "%Y-%m-%d").strftime("%d %b") for d in date_keys}

        long_rows = []
        for d, v in chart_data.items():
            for series, paise_amount in (("Confirmed", v["confirmed"]), ("Projected", v["projected"])):
                long_rows.append(
                    {
                        "date_label": date_labels[d],
                        "series": series,
                        "amount_rupees": float(from_paise(paise_amount)),
                        "amount_label": f"₹{format_inr(from_paise(paise_amount))}",
                    }
                )
        long_df = pl.DataFrame(long_rows)

        highlight = st.session_state[highlight_key]
        if highlight:
            selected_series = "Confirmed" if highlight == "confirmed" else "Projected"
            opacity_enc = alt.condition(
                alt.FieldEqualPredicate(field="series", equal=selected_series),
                alt.value(1.0),
                alt.value(0.22),
            )
        else:
            opacity_enc = alt.condition(alt.datum.series == "Projected", alt.value(0.62), alt.value(1.0))

        color_enc = alt.Color(
            "series:N",
            sort=["Confirmed", "Projected"],
            scale=alt.Scale(
                domain=["Confirmed", "Projected"],
                range=[tokens.ACTIVE_COLOR_SUCCESS, tokens.ACTIVE_COLOR_INFO],
            ),
            legend=alt.Legend(
                # Inset bottom-right: bars taper off toward later dates, so
                # this corner stays clear of the data. fillColor+padding give
                # it its own small card so it reads as a control, not a
                # floating label.
                title=None,
                orient="bottom-right",
                symbolType="square",
                symbolSize=90,
                labelFontSize=12,
                labelColor=tokens.MOCKUP_COLOR_TEXT_MUTED,
                fillColor=tokens.MOCKUP_COLOR_CARD_BACKGROUND,
                strokeColor=tokens.MOCKUP_COLOR_BORDER,
                cornerRadius=6,
                padding=10,
                offset=12,
            ),
        )
        shared_encodings = dict(
            x=alt.X(
                "date_label:N",
                title=None,
                sort=[date_labels[d] for d in date_keys],
                axis=alt.Axis(
                    domain=False, ticks=False, labelAngle=0, labelPadding=10, labelColor=tokens.MOCKUP_COLOR_TEXT_MUTED
                ),
            ),
            xOffset=alt.XOffset("series:N", sort=["Confirmed", "Projected"]),
        )

        bars = (
            alt.Chart(long_df.to_pandas())
            .mark_bar(
                cornerRadiusTopLeft=3,
                cornerRadiusTopRight=3,
                strokeWidth=1.25,
                width=alt.RelativeBandSize(0.82),
            )
            .encode(
                **shared_encodings,
                y=alt.Y(
                    "amount_rupees:Q",
                    title="Amount (₹)",
                    axis=alt.Axis(
                        # Indian numbering (L/Cr), not SI (k/M) -- d3-format has no
                        # native lakh/crore scale, so this is a Vega expression
                        # rather than a `format` specifier string.
                        labelExpr=_INDIAN_AXIS_LABEL_EXPR,
                        domain=False,
                        ticks=False,
                        gridColor=tokens.MOCKUP_COLOR_BORDER_SUBTLE,
                        labelColor=tokens.MOCKUP_COLOR_TEXT_MUTED,
                        titleColor=tokens.MOCKUP_COLOR_TEXT_MUTED,
                        titleFontWeight=500,
                    ),
                ),
                color=color_enc,
                stroke=color_enc,
                strokeDash=alt.condition(alt.datum.series == "Projected", alt.value([4, 3]), alt.value([1, 0])),
                opacity=opacity_enc,
                tooltip=[
                    alt.Tooltip("date_label:N", title="Date"),
                    alt.Tooltip("series:N", title="Status"),
                    alt.Tooltip("amount_label:N", title="Amount"),
                ],
            )
            .properties(height=460)
            .configure_view(strokeWidth=0)
            .configure_axis(labelFontSize=12, titleFontSize=12, labelFont=tokens.MOCKUP_FONT_TEXT, titleFont=tokens.MOCKUP_FONT_TEXT)
            .configure_legend(labelFont=tokens.MOCKUP_FONT_TEXT)
        )
        st.altair_chart(bars, use_container_width=True)

        # Replaces Vega-Embed's default action menu (hidden via theme.py's
        # .vega-embed rule) with the same grid-table component the Ledger
        # view uses.
        with st.expander("Show data"):
            table_html = ['<div class="forecast-data-table">']
            for label in ("Date", "Confirmed", "Projected"):
                right = "" if label == "Date" else " ledger-cell-right"
                table_html.append(f'<div class="ledger-cell ledger-cell-header{right}">{label}</div>')
            for d in date_keys:
                v = chart_data[d]
                table_html.append(f'<div class="ledger-cell">{date_labels[d]}</div>')
                table_html.append(
                    f'<div class="ledger-cell mono ledger-cell-right">₹{format_inr(from_paise(v["confirmed"]))}</div>'
                )
                table_html.append(
                    f'<div class="ledger-cell mono ledger-cell-right">₹{format_inr(from_paise(v["projected"]))}</div>'
                )
            table_html.append('</div>')
            st.markdown("".join(table_html), unsafe_allow_html=True)
    else:
        st.write(
            "This run is fully settled - nothing is currently projected. "
            "Trigger with a cutoff date to see an in-progress state."
        )
    st.markdown("</div>", unsafe_allow_html=True)


st.set_page_config(page_title="ZeroDrift", layout="wide")
inject_theme()
st.title("ZeroDrift - Reconciliation Dashboard")

if "runs" not in st.session_state:
    st.session_state.runs = {}  # batch_run_id -> display label
if "run_meta" not in st.session_state:
    st.session_state.run_meta = {}  # batch_run_id -> {"source", "seed", "cutoff", "triggered_at"/"loaded_at"}
if "view" not in st.session_state:
    st.session_state.view = "run"

has_runs = bool(st.session_state.runs)

with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:8px;padding-bottom:20px;border-bottom:1px solid #EEF0F3;margin-bottom:12px">'
        '<svg width="24" height="24" viewBox="0 0 240 240" xmlns="http://www.w3.org/2000/svg">'
        '<circle cx="120" cy="120" r="120" fill="#0B1F5B" />'
        '<text x="110" y="185" font-family="Inter, sans-serif" font-weight="800" font-size="170" fill="#FFFFFF" text-anchor="middle">Z</text>'
        '</svg>'
        '<div style="font-size:14px;font-weight:700;color:#12151C;letter-spacing:-0.01em">ZeroDrift</div>'
        '</div>',
        unsafe_allow_html=True,
    )
    for view_key in VIEWS:
        is_active = st.session_state.view == view_key
        disabled = (not has_runs) and view_key != "run"
        wrapper_key = f"nav-active-nav_{view_key}" if is_active else f"nav-inactive-nav_{view_key}"
        with st.container(key=wrapper_key):
            dot = "▸" if is_active else " "
            if st.button(f"{dot} {VIEW_LABELS[view_key]}", key=f"nav_{view_key}", disabled=disabled, use_container_width=True):
                st.session_state.view = view_key

view = st.session_state.view

if view == "run":
    active_source = st.session_state.get("trigger_source", "frozen")

    with st.container(key="run-card-frozen", border=True):
        st.subheader("Run the frozen benchmark")
        st.caption(
            "100 synthetic records, seed 42, committed to this repo - "
            "reproduce our numbers yourself."
        )
        if active_source != "frozen":
            st.caption('Switch Source to "frozen" below to use this card.')

    with st.container(key="run-card-seed", border=True):
        st.subheader("Bring your own seed")
        st.caption(
            "Generates a fresh, never-before-seen batch with the same "
            "category distribution. Full agent verification may take "
            "longer and calls a live model."
        )
        st.caption(
            f"Recommended: {RECOMMENDED_MAX_LIVE_SEED_RECORDS} records or fewer. A never-seen batch "
            "needs a real live model call per record needing judgment, against a hard daily token "
            "cap on this key - even the frozen dataset's own 100-record size would not fit a fresh "
            "day's budget if it weren't already fully cached. A larger batch fails cleanly with a "
            "clear message rather than posting partial results, but won't succeed until scaled down "
            "or retried on a later day."
        )
        if active_source != "seed":
            st.caption('Switch Source to "seed" below to use this card.')

    with st.container(key="run-form-card-config"):
        source = st.radio("Source", ["frozen", "seed"], key="trigger_source", horizontal=True)
        seed_value = None
        records_value = 100
        if source == "seed":
            seed_value = st.number_input("Seed", min_value=0, value=42, step=1, key="trigger_seed")
            records_value = st.number_input(
                "Records",
                min_value=1,
                value=100,
                step=1,
                key="trigger_records",
                help=(
                    f"Recommended {RECOMMENDED_MAX_LIVE_SEED_RECORDS} or fewer for a live seed batch - see "
                    "the daily token budget note above."
                ),
            )

        with st.expander("Advanced: cutoff date", expanded=False):
            st.caption(
                "Limits ledger settlement to a specific date - use this to see a "
                "genuine in-progress reconciliation state, e.g. for the forecast "
                "chart's confirmed vs. projected split."
            )
            gate_stage_2 = st.checkbox(
                "Limit settlement posting to a cutoff date",
                key="trigger_gate_stage_2",
            )
            trigger_as_of = None
            if gate_stage_2:
                # NOT min_value=date.today(): this cutoff selects a point
                # within the frozen dataset's own historical date range
                # (2025-01-06 to 2025-02-04), in the past relative to the
                # real calendar. A min_value=date.today() restriction would
                # silently clamp every real cutoff to today -- after the
                # whole dataset, gating nothing at all.
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
                    st.session_state.run_meta[new_id] = {
                        "source": source,
                        "seed": int(seed_value) if seed_value is not None else None,
                        "cutoff": trigger_as_of,
                        "triggered_at": datetime.now(),
                    }
                    st.session_state.last_loaded_id = new_id
                    st.success("Triggered batch run successfully!")
                    st.code(new_id, language=None)
                except api_client.ApiClientError as exc:
                    _caution_banner(f"Failed to trigger run: {exc.detail}")

    with st.container(key="run-form-card-load"):
        st.markdown("**Or view an existing run**")
        manual_id = st.text_input(
            "Run ID",
            key="manual_batch_run_id",
            placeholder="e.g. a1b2c3d4-5678-...",
            help="Paste a batch_run_id from a previous run to pull it into view without re-triggering it.",
        )
        if st.button("Load run", key="add_manual_button"):
            if not manual_id:
                _caution_banner("Enter a batch_run_id first.")
            else:
                try:
                    api_client.get_status(manual_id)  # validates the run exists
                    st.session_state.runs.setdefault(manual_id, f"Existing run · {manual_id[:8]}")
                    st.session_state.run_meta.setdefault(
                        manual_id, {"source": "existing", "seed": None, "cutoff": None, "loaded_at": datetime.now()}
                    )
                    st.session_state.last_loaded_id = manual_id
                    st.success("Added batch run successfully!")
                    st.code(manual_id, language=None)
                except api_client.ApiClientError as exc:
                    # exc.detail is already "unknown batch_run_id: <id>"
                    # (src/api/main.py::_require_known_run) -- prefixing
                    # "Unknown batch_run_id:" again would duplicate the
                    # phrase, so use the id the user typed instead.
                    st.error(f"Unknown batch_run_id: {manual_id}")

    if not st.session_state.runs:
        st.info("Trigger a batch run or add an existing batch_run_id to begin.")
    else:
        last_id = st.session_state.get("last_loaded_id") or next(iter(st.session_state.runs))
        with st.container(key="run-loaded-banner"):
            st.markdown(
                f'Run <span class="mono">{last_id}</span> loaded successfully.',
                unsafe_allow_html=True,
            )
            if st.button("View Overview ->", key="goto_overview_button"):
                st.session_state.view = "overview"
                # `view` is read once, above this block, so without an
                # immediate rerun this click would set the session-state
                # value but still render the Run screen on this same pass.
                # Sidebar nav buttons don't need this: their click happens
                # earlier in the script than the `view` read.
                st.rerun()

else:
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

    if not selected_runs:
        st.info("Select at least one run to view.")
        st.stop()

    as_of = st.session_state.get("forecast_as_of") or date.today()
    if view == "forecast":
        as_of = st.date_input("Forecast as-of date", value=date.today(), min_value=date.today(), key="forecast_as_of")

    columns = st.columns(len(selected_runs))
    for column, batch_run_id in zip(columns, selected_runs):
        with column:
            st.subheader(st.session_state.runs.get(batch_run_id, batch_run_id))
            _run_badge(batch_run_id)

            try:
                status = _load_status(batch_run_id)
            except api_client.ApiClientError as exc:
                _caution_banner(f"Could not load status: {exc.detail}")
                continue

            meta = st.session_state.run_meta.get(batch_run_id, {})
            try:
                if view == "overview":
                    _render_overview(batch_run_id, status, meta)
                elif view == "exceptions":
                    _render_exceptions(batch_run_id)
                elif view == "ledger":
                    _render_ledger(batch_run_id)
                elif view == "forecast":
                    _render_forecast(batch_run_id, as_of)
            except api_client.ApiClientError as exc:
                _caution_banner(f"Error loading view: {exc.detail}")
