"""Visual design tokens for the dashboard, sourced from @razorpay/blade
(MIT-licensed, github.com/razorpay/blade).

This dashboard's theme derives from Blade's published token source at
packages/blade/src/tokens/ -- it does not copy any Razorpay-branded asset,
logo, or wordmark. The Razorpay name and wordmark are trademark-protected
independently of the MIT license on the token/component code, and neither
appears anywhere in this UI; only the underlying, openly-licensed color,
spacing, radius, and typography values are reused.

Values below are read directly off Blade's `master` branch (fetched via the
GitHub API on 2026-09-01) and are not guessed approximations:

- Primary brand color: packages/blade/src/tokens/global/colors.ts,
  `chromatic.azure[500]` = hsla(218, 89%, 51%, 1) -- this is the color Blade's
  own light theme (theme/bladeTheme.ts) assigns to `action.background.primary`
  and `text.primary`. Converted to hex via standard HSL->RGB.
- Neutral grays: same file, `neutral.blueGrayLight` scale -- the scale Blade's
  light theme uses for backgrounds, borders, and body text.
- Spacing scale: packages/blade/src/tokens/global/spacing.ts (px values).
- Border radius: packages/blade/src/tokens/global/border.ts.
- Font stack: packages/blade/src/tokens/global/fontFamily/fontFamily.web.ts.
- Status colors: packages/blade/src/tokens/theme/bladeTheme.ts, the light
  theme's `feedback.text.{positive,information,notice}.intense` tokens --
  Blade's own semantic mapping for success/info/warning text, not a scale
  index picked by hand. positive = chromatic.emerald[700], information =
  chromatic.sapphire[700], notice = chromatic.cider[700].
"""
from __future__ import annotations

# -- Color --------------------------------------------------------------
# chromatic.azure[500], Blade's light-theme brand/primary action color.
COLOR_PRIMARY = "#1364F1"

# neutral.blueGrayLight[0] / [50] -- background surfaces.
COLOR_BACKGROUND = "#FFFFFF"
COLOR_BACKGROUND_SUBTLE = "#F7F7F7"

# neutral.blueGrayLight[200] / [300] -- border surfaces.
COLOR_BORDER_SUBTLE = "#DEE1E3"
COLOR_BORDER_NORMAL = "#C8CDD0"

# neutral.blueGrayLight[700] / [1300] -- text.
COLOR_TEXT = "#050505"
COLOR_TEXT_MUTED = "#616D75"

# -- Spacing (px), global/spacing.ts -------------------------------------
SPACING_1 = 2
SPACING_2 = 4
SPACING_3 = 8
SPACING_4 = 12
SPACING_5 = 16
SPACING_6 = 20
SPACING_7 = 24

# -- Radius (px), global/border.ts ---------------------------------------
RADIUS_SMALL = 8
RADIUS_MEDIUM = 12
RADIUS_LARGE = 16

# -- Status colors, theme/bladeTheme.ts feedback.text.*.intense ----------
COLOR_STATUS_POSITIVE = "#00753B"     # fast path -- feedback.text.positive.intense
COLOR_STATUS_INFORMATION = "#0070A8"  # agent resolved -- feedback.text.information.intense
COLOR_STATUS_NOTICE = "#C75300"       # honest exception -- feedback.text.notice.intense

# -- Typography, global/fontFamily/fontFamily.web.ts ---------------------
FONT_TEXT = '"Inter", "Inter Fallback Arial", Arial, sans-serif'
FONT_HEADING = '"TASA Orbiter", "TASA Orbiter Fallback Arial", Arial, sans-serif'
FONT_MONO = '"Menlo", "Roboto Mono", "Courier New", monospace'

# -- design.md follow-up (Layer 7 UI pass) --------------------------------
# The three values below are NOT separately-fetched Blade tokens -- they are
# CSS color-mix() blends of the already-sourced COLOR_STATUS_* intense
# tokens above, toward white, used for pill/card backgrounds and the caution
# banner. Kept as a plain CSS function (not a hand-picked new hex) so the
# only real color data in this file stays the originally verified fetch.
COLOR_STATUS_POSITIVE_SUBTLE = f"color-mix(in srgb, {COLOR_STATUS_POSITIVE} 12%, white)"
COLOR_STATUS_INFORMATION_SUBTLE = f"color-mix(in srgb, {COLOR_STATUS_INFORMATION} 12%, white)"
COLOR_STATUS_NOTICE_SUBTLE = f"color-mix(in srgb, {COLOR_STATUS_NOTICE} 12%, white)"

# Generic elevation shadow for the hover-lift effect (design.md 3.2.1) --
# not a Blade token (Blade's shadow scale wasn't part of the original
# fetch); a conventional, restrained low-elevation shadow.
SHADOW_HOVER = "0 4px 12px rgba(5, 5, 5, 0.10)"

# -- Mockup-replication pass ("ZeroDrift Dashboard.dc.html", user-supplied,
# read in full in the conversation this session continues) ----------------
# Everything below is copied directly from that file's <style> block and
# inline styles, NOT independently re-verified against Blade -- the user
# supplied this exact palette/spacing as the design of record for this pass
# and it deliberately overrides a few of the Blade-sourced values above
# where they conflict (most notably COLOR_STATUS_INFORMATION: the mockup
# uses a purple #5B4FCF for "Agent Resolved"/projected, not Blade's sapphire
# #0070A8). The overrides are applied explicitly below, not silently, so a
# reader can see both the original Blade fetch and what superseded it.
MOCKUP_COLOR_PAGE_BACKGROUND = "#F7F8FA"
MOCKUP_COLOR_CARD_BACKGROUND = "#FFFFFF"
MOCKUP_COLOR_BORDER = "#E4E7EC"
MOCKUP_COLOR_BORDER_SUBTLE = "#EEF0F3"
MOCKUP_COLOR_TEXT_DARK = "#12151C"
MOCKUP_COLOR_TEXT_MUTED = "#5B6472"
MOCKUP_COLOR_TEXT_FAINT = "#9AA3B2"
MOCKUP_COLOR_TEXT_DISABLED = "#C7CCD6"
MOCKUP_COLOR_LINK_HOVER = "#0E4FC2"

MOCKUP_COLOR_SUCCESS = "#1B7A43"       # fast path / confirmed cash
MOCKUP_COLOR_INFO_PURPLE = "#5B4FCF"   # agent resolved / projected cash -- REPLACES COLOR_STATUS_INFORMATION for this role
MOCKUP_COLOR_CAUTION_TEXT = "#8A5A00"  # honest exception / category pill text
MOCKUP_COLOR_CAUTION_BG = "#FEF7E8"
MOCKUP_COLOR_CAUTION_BORDER = "#F3DFAF"
MOCKUP_COLOR_SUCCESS_BANNER_BG = "#F0FBF4"
MOCKUP_COLOR_SUCCESS_BANNER_BORDER = "#C9EBD4"
MOCKUP_COLOR_RUN_BADGE_BG = "#F0F4FF"
MOCKUP_COLOR_RUN_BADGE_BORDER = "#DCE6FF"
MOCKUP_COLOR_NAV_ACTIVE_BG = "#F0F4FF"

MOCKUP_SIDEBAR_WIDTH_PX = 232

# The mockup's own font stack -- Inter (UI text) + Roboto Mono (numbers,
# tabular-nums), via Google Fonts, replacing the earlier Blade FONT_MONO
# stack (Menlo/Roboto Mono/Courier New) for anything styled ".mono" in this
# pass specifically.
MOCKUP_FONT_TEXT = "'Inter', system-ui, sans-serif"
MOCKUP_FONT_MONO = "'Roboto Mono', ui-monospace, Menlo, monospace"
MOCKUP_GOOGLE_FONTS_IMPORT = (
    "https://fonts.googleapis.com/css2?"
    "family=Inter:wght@400;500;600;700&family=Roboto+Mono:wght@400;500;600&display=swap"
)

# Effective status-color role bindings for this pass -- the mockup's values
# win over the earlier Blade fetch wherever they conflict (all three, not
# just information/purple: success and caution are also different literal
# hexes in the mockup). Everything in theme.py/app.py that colors the
# fast_path/agent_resolved/honest_exception roles uses these three, not the
# raw COLOR_STATUS_* constants above, so there is exactly one place this
# ever gets re-pointed.
ACTIVE_COLOR_SUCCESS = MOCKUP_COLOR_SUCCESS
ACTIVE_COLOR_INFO = MOCKUP_COLOR_INFO_PURPLE
ACTIVE_COLOR_CAUTION = MOCKUP_COLOR_CAUTION_TEXT
