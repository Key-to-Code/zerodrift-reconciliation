"""CSS injected for styling `.streamlit/config.toml` cannot reach (card
shadows, radius, table density). Values are pulled from tokens.py, which is
the single source of truth for both files -- see tokens.py's docstring for
where each value is sourced from Blade (github.com/razorpay/blade,
MIT-licensed; no Razorpay-branded asset, logo, or wordmark is used).
"""
from __future__ import annotations

import streamlit as st

from src.dashboard import tokens


def inject_theme() -> None:
    st.markdown(
        f"""
        <style>
        div[data-testid="stMetric"] {{
            background-color: {tokens.COLOR_BACKGROUND_SUBTLE};
            border-radius: {tokens.RADIUS_MEDIUM}px;
            padding: {tokens.SPACING_5}px;
            border: 1px solid {tokens.COLOR_BORDER_SUBTLE};
        }}

        div[data-testid="stDataFrame"] {{
            font-size: 0.9rem;
        }}

        html, body, [class*="css"] {{
            font-family: {tokens.FONT_TEXT};
        }}

        .status-resolved {{ color: {tokens.COLOR_STATUS_POSITIVE}; font-weight: 600; }}
        .status-agent-resolved {{ color: {tokens.COLOR_STATUS_INFORMATION}; font-weight: 600; }}
        .status-honest-exception {{ color: {tokens.COLOR_STATUS_NOTICE}; font-weight: 600; }}

        .money-figure {{
            font-family: {tokens.FONT_MONO};
            font-variant-numeric: tabular-nums;
        }}

        /* -- design.md follow-up (Layer 7 UI pass) ------------------------ */

        /* Run screen: two trigger cards + the cutoff expander/date control. */
        [class*="st-key-run-card-"] {{
            background-color: {tokens.COLOR_BACKGROUND};
            border: 1px solid {tokens.COLOR_BORDER_SUBTLE};
            border-radius: {tokens.RADIUS_MEDIUM}px;
            padding: {tokens.SPACING_6}px;
        }}
        [class*="st-key-trigger_as_of"] input, [class*="st-key-forecast_as_of"] input {{
            border-radius: {tokens.RADIUS_SMALL}px;
            border: 1px solid {tokens.COLOR_PRIMARY};
        }}

        /* Overview: accent-left-border metric cards with a hover lift, and
           hovering a stacked-bar segment highlights its matching card --
           pure CSS (:has()), no JavaScript. Each status's card and bar
           segment share the "metric-card-<status>"/"bar-segment-<status>"
           name fragment so one selector covers every run's column. */
        [class*="metric-card-"] div[data-testid="stMetric"] {{
            transition: box-shadow 0.15s ease, transform 0.15s ease;
        }}
        [class*="metric-card-"] div[data-testid="stMetric"]:hover {{
            box-shadow: {tokens.SHADOW_HOVER};
            transform: translateY(-2px);
        }}
        [class*="metric-card-positive"] div[data-testid="stMetric"] {{
            border-left: 4px solid {tokens.COLOR_STATUS_POSITIVE};
        }}
        [class*="metric-card-information"] div[data-testid="stMetric"] {{
            border-left: 4px solid {tokens.COLOR_STATUS_INFORMATION};
        }}
        [class*="metric-card-notice"] div[data-testid="stMetric"] {{
            border-left: 4px solid {tokens.COLOR_STATUS_NOTICE};
        }}

        .stacked-bar {{
            display: flex;
            width: 100%;
            height: 28px;
            border-radius: {tokens.RADIUS_SMALL}px;
            overflow: hidden;
            margin: {tokens.SPACING_3}px 0 {tokens.SPACING_5}px 0;
        }}
        .bar-segment-positive {{ background-color: {tokens.COLOR_STATUS_POSITIVE}; }}
        .bar-segment-information {{ background-color: {tokens.COLOR_STATUS_INFORMATION}; }}
        .bar-segment-notice {{ background-color: {tokens.COLOR_STATUS_NOTICE}; }}
        .stacked-bar > div {{
            transition: opacity 0.15s ease;
        }}
        [class*="overview-section-"]:has(.bar-segment-positive:hover) [class*="metric-card-positive"] div[data-testid="stMetric"],
        [class*="overview-section-"]:has(.bar-segment-information:hover) [class*="metric-card-information"] div[data-testid="stMetric"],
        [class*="overview-section-"]:has(.bar-segment-notice:hover) [class*="metric-card-notice"] div[data-testid="stMetric"] {{
            box-shadow: {tokens.SHADOW_HOVER};
            transform: translateY(-2px);
        }}

        /* Exceptions: category pill. All categories share one caution tint
           (design.md 3.3.2) -- the pill communicates "needs a human," not
           which category, so one color is correct, not a missing feature. */
        .category-pill {{
            display: inline-block;
            background-color: {tokens.COLOR_STATUS_NOTICE_SUBTLE};
            color: {tokens.COLOR_STATUS_NOTICE};
            border-radius: 999px;
            padding: 2px 10px;
            font-size: 0.75rem;
            font-weight: 600;
        }}

        /* Ledger: TOTAL row treatment and the balanced-net checkmark. */
        .ledger-total-row {{
            background-color: {tokens.COLOR_BACKGROUND_SUBTLE};
            font-weight: 700;
            border-top: 2px solid {tokens.COLOR_TEXT};
        }}
        .balanced-badge {{
            color: {tokens.COLOR_STATUS_POSITIVE};
            font-weight: 700;
        }}

        /* Errors: a calm caution banner instead of Streamlit's default red
           st.error box (design.md 4, "never a red error box"). */
        .caution-banner {{
            background-color: {tokens.COLOR_STATUS_NOTICE_SUBTLE};
            border: 1px solid {tokens.COLOR_STATUS_NOTICE};
            border-radius: {tokens.RADIUS_SMALL}px;
            padding: {tokens.SPACING_4}px {tokens.SPACING_5}px;
            color: {tokens.COLOR_TEXT};
        }}

        /* Forecast: centered content column. */
        .centered-content {{
            max-width: 900px;
            margin-left: auto;
            margin-right: auto;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
