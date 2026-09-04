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
        @import url('{tokens.MOCKUP_GOOGLE_FONTS_IMPORT}');

        /* -- Mockup-replication pass: page-level base, from the mockup's own
           <style> block almost verbatim. ------------------------------- */
        .stApp {{
            background-color: {tokens.MOCKUP_COLOR_PAGE_BACKGROUND};
        }}
        html, body, [class*="css"] {{
            font-family: {tokens.MOCKUP_FONT_TEXT} !important;
        }}
        .mono, .money-figure {{
            font-family: {tokens.MOCKUP_FONT_MONO};
            font-variant-numeric: tabular-nums;
        }}
        a {{ color: {tokens.COLOR_PRIMARY}; }}
        a:hover {{ color: {tokens.MOCKUP_COLOR_LINK_HOVER}; }}

        /* Sidebar: 232px, white, right border -- matches the mockup's
           fixed-width nav rail. Streamlit's own sidebar resize handle still
           works; this only sets the resting width/look. */
        section[data-testid="stSidebar"] {{
            width: {tokens.MOCKUP_SIDEBAR_WIDTH_PX}px;
            background-color: {tokens.MOCKUP_COLOR_CARD_BACKGROUND};
            border-right: 1px solid {tokens.MOCKUP_COLOR_BORDER};
        }}

        /* Nav buttons: the active view (session_state.view) gets the
           active-nav treatment; disabled (no run loaded yet) gets the
           mockup's faint disabled text color instead of Streamlit's default
           greyed-out look. */
        /* Professional Sidebar */
        section[data-testid="stSidebar"] {{
            background-color: {tokens.COLOR_BACKGROUND} !important;
            border-right: 1px solid {tokens.COLOR_BORDER_SUBTLE};
            box-shadow: 2px 0 10px rgba(0,0,0,0.02);
        }}
        [class*="st-key-nav_"] button {{
            width: 100%;
            text-align: left;
            background: transparent;
            border: none;
            color: {tokens.MOCKUP_COLOR_TEXT_DARK};
            font-weight: 500;
            padding: 12px 20px;
            margin-bottom: 4px;
            border-radius: {tokens.RADIUS_MEDIUM}px;
            transition: all 0.2s ease;
        }}
        [class*="st-key-nav_"] button:disabled {{
            color: {tokens.MOCKUP_COLOR_TEXT_DISABLED};
        }}
        [class*="st-key-nav_"] button:not(:disabled):hover {{
            background-color: {tokens.MOCKUP_COLOR_PAGE_BACKGROUND};
            transform: translateX(4px);
        }}
        [class*="st-key-nav-active-"] button {{
            background-color: {tokens.MOCKUP_COLOR_NAV_ACTIVE_BG} !important;
            color: {tokens.COLOR_PRIMARY};
            font-weight: 600;
        }}

        /* Persistent run badge, shown on every non-Run screen once a run is
           loaded -- mirrors the mockup's "Viewing run <id>" bar. */
        .run-badge-bar {{
            background-color: {tokens.MOCKUP_COLOR_RUN_BADGE_BG};
            border: 1px solid {tokens.MOCKUP_COLOR_RUN_BADGE_BORDER};
            border-radius: {tokens.RADIUS_SMALL}px;
            padding: {tokens.SPACING_3}px {tokens.SPACING_5}px;
            color: {tokens.MOCKUP_COLOR_TEXT_MUTED};
            font-size: 0.85rem;
            margin-bottom: {tokens.SPACING_5}px;
        }}
        .run-badge-bar .mono {{
            background: {tokens.MOCKUP_COLOR_CARD_BACKGROUND};
            border: 1px solid {tokens.MOCKUP_COLOR_RUN_BADGE_BORDER};
            border-radius: 4px;
            padding: 2px 7px;
            color: {tokens.COLOR_PRIMARY};
            font-weight: 600;
        }}

        /* Success banner (run loaded), mirrors the mockup's green confirm
           box around the "View Overview ->" button. */
        [class*="st-key-run-loaded-banner"] {{
            background-color: {tokens.MOCKUP_COLOR_SUCCESS_BANNER_BG};
            border: 1px solid {tokens.MOCKUP_COLOR_SUCCESS_BANNER_BORDER};
            border-radius: {tokens.RADIUS_MEDIUM}px;
            padding: {tokens.SPACING_5}px {tokens.SPACING_6}px;
        }}

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

        /* Run screen: two trigger cards + form configuration cards. */
        [class*="st-key-run-card-"], [class*="st-key-run-form-card"] {{
            background-color: {tokens.COLOR_BACKGROUND};
            border: 1px solid {tokens.COLOR_BORDER_SUBTLE};
            border-radius: {tokens.RADIUS_MEDIUM}px;
            padding: {tokens.SPACING_6}px;
            margin-bottom: {tokens.SPACING_6}px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.03);
            transition: box-shadow 0.2s ease, transform 0.2s ease, border-color 0.2s ease;
        }}
        [class*="st-key-run-card-"]:hover, [class*="st-key-run-form-card"]:hover {{
            box-shadow: 0 4px 12px rgba(0,0,0,0.06);
            transform: translateY(-2px);
            border-color: {tokens.MOCKUP_COLOR_BORDER};
        }}
        /* Decorative Radio buttons - Segemented Pill style */
        div[data-testid="stRadio"] > div[role="radiogroup"] {{
            display: flex;
            gap: 8px;
            background-color: {tokens.MOCKUP_COLOR_PAGE_BACKGROUND};
            padding: 6px;
            border-radius: 99px;
            width: fit-content;
            border: 1px solid {tokens.MOCKUP_COLOR_BORDER};
        }}
        /* Hide the actual radio circle from Streamlit */
        div[data-testid="stRadio"] label > div:first-child {{
            display: none !important;
        }}
        div[data-testid="stRadio"] label {{
            background-color: transparent;
            padding: 8px 18px;
            border-radius: 99px;
            cursor: pointer;
            transition: all 0.25s ease;
            margin: 0 !important;
            border: 1px solid transparent;
        }}
        div[data-testid="stRadio"] label:hover {{
            background-color: {tokens.COLOR_BACKGROUND};
            box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        }}
        div[data-testid="stRadio"] label:has(input[type="radio"]:checked) {{
            background-color: {tokens.COLOR_BACKGROUND};
            border: 1px solid {tokens.COLOR_PRIMARY};
            box-shadow: 0 2px 8px {tokens.COLOR_PRIMARY}30;
        }}
        div[data-testid="stRadio"] input[type="radio"]:checked + div {{
            font-weight: 600;
            color: {tokens.COLOR_PRIMARY};
        }}
        
        /* Expander Styling */
        div[data-testid="stExpander"] {{
            border: 1px solid {tokens.COLOR_BORDER_SUBTLE};
            border-radius: {tokens.RADIUS_MEDIUM}px;
            background-color: {tokens.MOCKUP_COLOR_PAGE_BACKGROUND};
            box-shadow: 0 1px 2px rgba(0,0,0,0.02);
            overflow: hidden;
            transition: all 0.2s ease;
        }}
        div[data-testid="stExpander"]:hover {{
            box-shadow: 0 2px 6px rgba(0,0,0,0.04);
        }}
        div[data-testid="stExpander"] summary {{
            padding: 12px 16px;
            background-color: transparent;
            font-weight: 600;
        }}
        div[data-testid="stExpanderDetails"] {{
            padding: 16px 24px;
        }}

        /* Fix Streamlit Popover Dropdowns (Multiselect/Select) */
        div[data-baseweb="popover"] > div {{
            background-color: {tokens.COLOR_BACKGROUND} !important;
            border: 1px solid {tokens.COLOR_BORDER_SUBTLE} !important;
            border-radius: {tokens.RADIUS_MEDIUM}px !important;
            box-shadow: 0 4px 16px rgba(0,0,0,0.12) !important;
        }}

        /* Premium Checkbox Toggle */
        div[data-testid="stCheckbox"] label > div:first-child {{
            border-radius: 99px;
            width: 36px;
            height: 20px;
            background-color: {tokens.COLOR_BORDER_SUBTLE};
            border: none;
            position: relative;
            transition: all 0.25s ease;
        }}
        div[data-testid="stCheckbox"] label > div:first-child::after {{
            content: '';
            position: absolute;
            top: 2px;
            left: 2px;
            width: 16px;
            height: 16px;
            background-color: white;
            border-radius: 50%;
            transition: all 0.25s cubic-bezier(0.4, 0.0, 0.2, 1);
            box-shadow: 0 1px 3px rgba(0,0,0,0.15);
        }}
        div[data-testid="stCheckbox"] input:checked + div {{
            background-color: {tokens.COLOR_PRIMARY};
        }}
        div[data-testid="stCheckbox"] input:checked + div::after {{
            transform: translateX(16px);
        }}
        div[data-testid="stCheckbox"] label > div:first-child svg {{
            display: none !important;
        }}

        /* Date Picker Styling */
        div[data-testid="stDateInput"] > div {{
            background-color: {tokens.COLOR_BACKGROUND};
            border-radius: {tokens.RADIUS_SMALL}px;
            border: 1px solid {tokens.COLOR_BORDER_SUBTLE};
            transition: border-color 0.2s ease, box-shadow 0.2s ease;
            padding: 2px;
        }}
        div[data-testid="stDateInput"] > div:focus-within {{
            border-color: {tokens.COLOR_PRIMARY};
            box-shadow: 0 0 0 3px {tokens.COLOR_PRIMARY}20;
        }}
        [class*="st-key-trigger_as_of"] input, [class*="st-key-forecast_as_of"] input {{
            font-family: {tokens.MOCKUP_FONT_MONO};
            padding: 8px 12px;
            border: none !important;
            box-shadow: none !important;
        }}

        /* Overview: accent-left-border metric cards with a hover lift, and
           hovering a stacked-bar segment highlights its matching card --
           pure CSS (:has()), no JavaScript. Each status's card and bar
           segment share the "metric-card-<status>"/"bar-segment-<status>"
           name fragment so one selector covers every run's column. */
        [class*="metric-card-"] div[data-testid="stMetric"] {{
            background-color: {tokens.COLOR_BACKGROUND};
            border: 1px solid {tokens.COLOR_BORDER_SUBTLE};
            border-radius: {tokens.RADIUS_MEDIUM}px;
            padding: {tokens.SPACING_4}px {tokens.SPACING_6}px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.03);
            transition: box-shadow 0.2s ease, transform 0.2s ease, border-color 0.2s ease;
        }}
        [class*="metric-card-"] div[data-testid="stMetric"]:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.06);
        }}
        [class*="metric-card-positive"] div[data-testid="stMetric"] {{
            border-left: 4px solid {tokens.ACTIVE_COLOR_SUCCESS};
        }}
        [class*="metric-card-positive"] div[data-testid="stMetric"]:hover {{
            border-color: {tokens.ACTIVE_COLOR_SUCCESS};
            box-shadow: 0 4px 12px {tokens.ACTIVE_COLOR_SUCCESS}40;
        }}
        [class*="metric-card-information"] div[data-testid="stMetric"] {{
            border-left: 4px solid {tokens.ACTIVE_COLOR_INFO};
        }}
        [class*="metric-card-information"] div[data-testid="stMetric"]:hover {{
            border-color: {tokens.ACTIVE_COLOR_INFO};
            box-shadow: 0 4px 12px {tokens.ACTIVE_COLOR_INFO}40;
        }}
        [class*="metric-card-notice"] div[data-testid="stMetric"] {{
            border-left: 4px solid {tokens.ACTIVE_COLOR_CAUTION};
        }}
        [class*="metric-card-notice"] div[data-testid="stMetric"]:hover {{
            border-color: {tokens.ACTIVE_COLOR_CAUTION};
            box-shadow: 0 4px 12px {tokens.ACTIVE_COLOR_CAUTION}40;
        }}

        .stacked-bar {{
            display: flex;
            width: 100%;
            height: 28px;
            border-radius: {tokens.RADIUS_SMALL}px;
            overflow: hidden;
            margin: {tokens.SPACING_3}px 0 {tokens.SPACING_5}px 0;
            border: 1px solid {tokens.MOCKUP_COLOR_BORDER};
        }}
        .bar-segment-positive {{ background-color: {tokens.ACTIVE_COLOR_SUCCESS}; }}
        .bar-segment-information {{ background-color: {tokens.ACTIVE_COLOR_INFO}; }}
        .bar-segment-notice {{ background-color: {tokens.ACTIVE_COLOR_CAUTION}; }}
        .stacked-bar > div {{
            transition: opacity 0.15s ease;
        }}
        [class*="overview-section-"]:has(.bar-segment-positive:hover) [class*="metric-card-positive"] div[data-testid="stMetric"] {{
            box-shadow: 0 4px 12px {tokens.ACTIVE_COLOR_SUCCESS}40;
            border-color: {tokens.ACTIVE_COLOR_SUCCESS};
            transform: translateY(-2px);
        }}
        [class*="overview-section-"]:has(.bar-segment-information:hover) [class*="metric-card-information"] div[data-testid="stMetric"] {{
            box-shadow: 0 4px 12px {tokens.ACTIVE_COLOR_INFO}40;
            border-color: {tokens.ACTIVE_COLOR_INFO};
            transform: translateY(-2px);
        }}
        [class*="overview-section-"]:has(.bar-segment-notice:hover) [class*="metric-card-notice"] div[data-testid="stMetric"] {{
            box-shadow: 0 4px 12px {tokens.ACTIVE_COLOR_CAUTION}40;
            border-color: {tokens.ACTIVE_COLOR_CAUTION};
            transform: translateY(-2px);
        }}
        
        /* Highlight bar segment when hovering its card */
        [class*="overview-section-"]:has([class*="metric-card-"] div[data-testid="stMetric"]:hover) .stacked-bar > div {{
            opacity: 0.3;
        }}
        [class*="overview-section-"]:has([class*="metric-card-positive"] div[data-testid="stMetric"]:hover) .bar-segment-positive {{
            opacity: 1 !important;
            filter: brightness(1.15);
        }}
        [class*="overview-section-"]:has([class*="metric-card-information"] div[data-testid="stMetric"]:hover) .bar-segment-information {{
            opacity: 1 !important;
            filter: brightness(1.15);
        }}
        [class*="overview-section-"]:has([class*="metric-card-notice"] div[data-testid="stMetric"]:hover) .bar-segment-notice {{
            opacity: 1 !important;
            filter: brightness(1.15);
        }}

        /* Exceptions: category pill. All categories share one caution tint
           (design.md 3.3.2) -- the pill communicates "needs a human," not
           which category, so one color is correct, not a missing feature. */
        .category-pill {{
            display: inline-block;
            background-color: {tokens.MOCKUP_COLOR_CAUTION_BG};
            color: {tokens.MOCKUP_COLOR_CAUTION_TEXT};
            border-radius: 999px;
            padding: 4px 10px;
            margin-bottom: {tokens.SPACING_4}px;
            font-size: 0.7rem;
            font-weight: 500;
        }}

        /* Ledger: custom HTML grid table (not st.dataframe -- a Styler
           can't render the TOTAL row's checkmark badge). Column widths
           mirror the mockup's grid-template-columns. */
        .ledger-table {{
            background-color: {tokens.MOCKUP_COLOR_CARD_BACKGROUND};
            border: 1px solid {tokens.MOCKUP_COLOR_BORDER};
            border-radius: {tokens.RADIUS_MEDIUM}px;
            overflow: hidden;
            font-size: 0.85rem;
        }}
        .ledger-row {{
            display: grid;
            grid-template-columns: 1.5fr 2.5fr 1fr 1.2fr 1.2fr 1.2fr;
            gap: 16px;
            padding: 12px 20px;
            border-bottom: 1px solid {tokens.MOCKUP_COLOR_BORDER_SUBTLE};
            color: {tokens.MOCKUP_COLOR_TEXT_DARK};
            align-items: center;
        }}
        .ledger-header-row {{
            background-color: {tokens.MOCKUP_COLOR_PAGE_BACKGROUND};
            font-size: 0.68rem;
            font-weight: 600;
            color: {tokens.MOCKUP_COLOR_TEXT_MUTED};
            text-transform: uppercase;
            letter-spacing: 0.03em;
            border-bottom: 1px solid {tokens.MOCKUP_COLOR_BORDER};
        }}
        .ledger-total-row {{
            background-color: {tokens.MOCKUP_COLOR_PAGE_BACKGROUND};
            font-weight: 700;
            border-top: 2px solid {tokens.MOCKUP_COLOR_TEXT_DARK};
            border-bottom: none;
            color: {tokens.MOCKUP_COLOR_TEXT_DARK};
        }}
        .ledger-cell-right {{ text-align: right; }}
        .balanced-badge {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 16px;
            height: 16px;
            background-color: {tokens.ACTIVE_COLOR_SUCCESS};
            color: {tokens.MOCKUP_COLOR_CARD_BACKGROUND};
            border-radius: 50%;
            font-size: 10px;
            margin-right: 6px;
        }}

        /* Errors: a calm caution banner instead of Streamlit's default red
           st.error box (design.md 4, "never a red error box"). */
        .caution-banner {{
            background-color: {tokens.MOCKUP_COLOR_CAUTION_BG};
            border: 1px solid {tokens.MOCKUP_COLOR_CAUTION_BORDER};
            border-radius: {tokens.RADIUS_SMALL}px;
            padding: {tokens.SPACING_4}px {tokens.SPACING_5}px;
            margin-top: 8px;
            margin-bottom: 12px;
            color: {tokens.MOCKUP_COLOR_CAUTION_TEXT};
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
