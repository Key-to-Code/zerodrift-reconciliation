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
        </style>
        """,
        unsafe_allow_html=True,
    )
