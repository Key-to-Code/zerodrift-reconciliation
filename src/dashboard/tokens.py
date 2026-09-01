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
