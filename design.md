# ZeroDrift — Design Specification
### Read alongside `docs/plan.md` and `CLAUDE.md`. This file governs `src/dashboard/` — visual design, layout, interaction, and performance. Section 3.3's exception-trace requirement needs one small, explicitly-scoped backend addition (see 3.3, "Reasoning trace data"); everything else changes no backend behavior — every number still comes from the API exactly as Layer 6/7 already built it.

---

## 0. Design principles — the rules everything else here follows

1. **The UI's job is to make the architecture visible, not to decorate it.** Every screen exists to answer one specific question a judge or user is asking. If an element doesn't serve that question, it doesn't belong on that screen.
2. **Calm, not flashy.** This is a finance product. Financial software that looks playful or trend-chasing reads as untrustworthy to the audience that matters here. Restraint is the differentiator, not vibrancy. Interaction affordances (hover states, transitions — see Section 3) are subtle: small, soft-shadow lifts, never bounce, spring, or springy easing.
3. **Numbers are facts, treated visually as facts.** Monospaced, right-aligned, unambiguous. A percentage is always shown with its underlying count nearby (the percentage can be visually primary — larger, bolder — with the count secondary and smaller below it, but never drop the count entirely: "72%" alone doesn't tell a reader whether that's 72 of 100 or 72 of 10,000, and this project's credibility rests on traceable exact numbers).
4. **Nothing is fabricated visually either.** If data is loading, say so. If a section has no data, show an explicit empty state, never a blank gap that looks broken. If the UI wants to show something the API doesn't yet return, that's a signal to add the field, not to approximate it from unrelated text.
5. **Fast perceived response beats fast actual response.** Streamlit reruns the whole script on most interactions — the design must account for this explicitly (Section 5), not fight it after the fact.
6. **"Professional," not "borrowed."** The target feel is polished, modern fintech software — comparable in quality to well-designed dashboard products generally — achieved through the specific, concrete treatments in this document (real shadows, real hover states, real typographic hierarchy), not by copying any single named product's visual identity wholesale. If a specific pattern from a specific site is worth referencing, name the pattern explicitly when it comes up, rather than "make it look like X" as a blanket instruction — that's how visual mimicry problems happen, and it also just doesn't give an implementer anything concrete to build.

---

## 1. Visual language

### 1.1 Color — semantic, not decorative

Every color on screen must mean something. No color exists purely for visual interest.

| Token | Use | Value source |
|---|---|---|
| `COLOR_PRIMARY` | Primary actions, active nav state, links | Blade `chromatic.azure[500]` (`#1364F1`) |
| `COLOR_SUCCESS` | Fast-path resolved, confirmed cash | Blade `feedback.text.positive.intense` |
| `COLOR_INFO` | Agent-resolved, projected cash | Blade `feedback.text.information.intense` |
| `COLOR_CAUTION` | Honest exception | Blade `feedback.text.notice.intense` — **never red**. Red reads as "broken." Honest refusal is a correct, calm outcome and must look like one. |
| `COLOR_NEUTRAL_900` | Primary text | Blade neutral scale, darkest |
| `COLOR_NEUTRAL_500` | Secondary text, captions, timestamps | Blade neutral scale, mid |
| `COLOR_NEUTRAL_100` | Card/table backgrounds, zebra striping | Blade neutral scale, lightest |
| `COLOR_BORDER` | Card borders, table dividers | Blade neutral scale, just above background |

**Rule:** the three-way resolution split (fast_path / agent_resolved / honest_exception) uses `COLOR_SUCCESS` / `COLOR_INFO` / `COLOR_CAUTION` consistently everywhere it appears — summary cards, exception list, chart legends, badges. Learn the code once, it never means something different elsewhere. The Forecast screen's confirmed/projected pair reuses `COLOR_SUCCESS`/`COLOR_INFO` for the same reason — no third, unrelated palette gets introduced for that screen.

### 1.2 Typography

- **Body/UI text:** Blade's `Inter` stack.
- **Numbers — all of them, everywhere:** a monospace face (Blade's `Menlo` fallback, or `Roboto Mono`) with `font-variant-numeric: tabular-nums`. Non-negotiable for a finance product — numbers in a proportional font visually jitter across rows, undermining trust in the trial balance specifically.
- **Type scale:** 3 sizes only. Section headers (~1.25rem, semibold), body/table text (~0.9rem, regular), captions/metadata (~0.75rem, `COLOR_NEUTRAL_500`). Hierarchy comes from weight and color, not a fourth size.

### 1.3 Spacing, density, and surface treatment

- Base unit: Blade's spacing scale.
- **Cards, not borders-everywhere.** Group related content in a card (`COLOR_NEUTRAL_100` background, `COLOR_BORDER` 1px border, Blade's border-radius) rather than separating everything with horizontal rules.
- **Give containers real depth.** A flat white background with thin borders and no shadow is the single biggest tell of an unstyled page. Cards and table containers get a soft, subtle box-shadow (small blur radius, low opacity — e.g. `0 1px 3px rgba(0,0,0,0.06)`) so surfaces read as physically distinct layers, not just outlined regions.
- **Tables are dense by default** — this is a daily-use analyst tool, not a marketing page. Row height fits ~15 rows without scrolling on a normal laptop screen.
- **Table-specific treatment** (applies to Exceptions and Ledger screens): subtle zebra striping (`COLOR_NEUTRAL_100` on alternating rows); header row visually distinct via weight and background, not just a bottom border; row hover state (a faint background tint) on any row that's interactive (expandable); the table container itself gets rounded corners and the shadow treatment above, so it reads as one designed surface rather than a raw HTML table dropped onto the page.

### 1.4 Iconography

Minimal. Icons only for status (three resolution states) and navigation. No decorative icons. Blade's icon set if available; otherwise small inline status markers (✓ / ⚙ / ⚠), never large or standalone.

### 1.5 Interaction affordances

- **Hover states are real but restrained.** Metric cards and forecast summary cards lift slightly on hover (2–4px translate, soft shadow increase) — a subtle physical response, not a bounce or scale animation.
- **Linked hover, where specified below**, connects related elements across a screen (e.g., hovering a chart segment highlighting its corresponding card) so the interface visibly demonstrates that the summary numbers and the detailed view are the same data, not coincidentally similar.

---

## 2. Information architecture — what exists, and where

The dashboard has **one entry screen (Run) plus four content views** (Overview / Exceptions / Ledger / Forecast). Run is not itself a "view" in the same sense as the other four — it's how a `batch_run_id` gets selected in the first place; the four content views are peer screens a user moves between once a run is loaded.

```
┌─────────────────┐
│  ZeroDrift       │  <- wordmark, small, top of sidebar
├─────────────────┤
│  Run             │  <- entry point: trigger or select a run
├─────────────────┤
│  ▸ Overview      │  <- match-rate summary, the headline screen
│    Exceptions    │  <- honest exception list
│    Ledger        │  <- trial balance
│    Forecast      │  <- cash projection chart
└─────────────────┘
```

Visually, "Run" can sit slightly separated from the four content views (a divider, or simply first in the list before a rule) to reinforce that it's a different kind of screen, not a fifth peer view.

**Run selection is global, not per-page.** The currently-selected `batch_run_id` persists across all four content views via `st.session_state`, shown as a small persistent badge at the top of the main content area on every screen (e.g., "Viewing run `a1b2c3d4…` — frozen dataset — triggered 2 min ago"). A user should never lose track of which run they're looking at while navigating.

**Compare mode.** A "Compare runs" toggle lives on the Run screen. Off by default (single-run view, as described throughout this document). When enabled, it switches all four content views to the existing side-by-side column layout for two or more selected runs, reusing the comparison mechanism already built and tested in Layer 7 (tests 13/20/21) — that logic is not rebuilt, only gated behind this toggle instead of being permanently on. Single-run view stays the default because most real interactions concern one run at a time, and a permanently split-column layout would work against Section 0's "calm" principle for the common case.

---

## 3. Screen-by-screen specification

### 3.1 Run — trigger and selection

**Purpose:** get a `batch_run_id` into view, either by triggering a new run or pulling up an existing one. This is the entry point; everything else depends on a run being selected.

**Layout, top to bottom:**

1. **Two side-by-side option cards** (not a dropdown-first layout) — the first thing a judge sees should visually communicate "there are two ways to prove this system, pick one":
   - **Card A — "Run the frozen benchmark."** One button: `Trigger frozen batch`. Subtext: "100 synthetic records, seed 42, committed to this repo — reproduce our numbers yourself." The safe, fast, always-works option.
   - **Card B — "Bring your own seed."** A numeric input for a seed value, one button: `Trigger live batch`. Subtext: "Generates a fresh, never-before-seen batch with the same category distribution. Full agent verification may take longer and calls a live model." The credibility move — explicitly flagged as slower/live so nobody is surprised by latency.

2. **Below both cards, collapsed by default:** an expander labeled "Advanced: cutoff date." Inside: a custom-styled date-range control (see "Date picker" below) plus a one-line explanation ("Limits ledger settlement to a specific date — use this to see a genuine in-progress reconciliation state, e.g. for the forecast chart's confirmed vs. projected split"). Collapsed by default — it's a demo/debugging tool, not a primary action.

   **Date picker.** Streamlit's default `st.date_input` reads as an unstyled native browser control and stands out against the rest of the page. Build a CSS-skinned version instead: the same underlying `st.date_input` widget, restyled via the project's existing CSS-injection mechanism (`theme.py`) to match the token palette (rounded container, `COLOR_BORDER`, `COLOR_PRIMARY` accent on the selected date) — a genuinely custom-*looking* picker without a custom-*built* component, keeping this inside Section 6's "no custom JavaScript" boundary.

3. **Below that, always visible:** "Or view an existing run." Label the field with a plain-language placeholder — no example UUID shown (a raw `batch_id`-shaped placeholder reads as a debug tool, not a product). Use something like *"Paste a run ID to view it"*, with a small helper caption below the field for anyone who wants the format explained, rather than an inline fake ID as the placeholder text itself. One button, `Load run`. Clean, inline error message on an unknown ID (reusing the API's actual 404 — see Section 4), styled as a caution-colored inline banner, never a red error box or browser-style alert.

4. **Once a run is loaded:** the persistent run badge (Section 2) appears, and a "View Overview →" button advances to the next screen. Don't auto-navigate — let the user see confirmation the run loaded successfully first.

**Loading state:** while a run is triggering, replace the button with a spinner and text that differs by source — "Posting to ledger…" (frozen) vs. "Calling agent, this may take a minute…" (live seed) — since the honest expected wait time genuinely differs and hiding that is a small dishonesty.

### 3.2 Overview — the headline screen

**Purpose:** in one glance, without scrolling, show the architecture's entire story: how much resolved automatically, how much needed judgment, how much was honestly refused.

**Layout:**

1. **Three large metric cards in a row**, equal width, using `st.metric` (existing tests assert on this — don't relitigate it):
   - Fast Path Resolved — percentage primary/large, count secondary/small below it, `COLOR_SUCCESS` accent border-left (4px)
   - Agent Resolved — same treatment, `COLOR_INFO` accent border-left
   - Honest Exceptions — same treatment, `COLOR_CAUTION` accent border-left
   
   Cards lift subtly on hover per Section 1.5 — they are not static/stagnant elements.

   Directly below the three cards, one full-width horizontal **stacked bar** showing the same three counts as proportional segments in the same three colors — the "get it in one glance" element. **Hovering a segment of the bar visually raises (Section 1.5's hover-lift) its corresponding metric card above it**, so the connection between the summary bar and the detail cards is demonstrated through interaction, not just implied by shared color.

2. **Below the bar, a small caption line**, `COLOR_NEUTRAL_500`, computed from the real API response (never hardcoded numbers): *"63 resolved by deterministic matching alone — no model call. 20 required agent judgment. 17 could not be confidently resolved and were routed to suspense rather than guessed."*

3. **A secondary row, smaller cards:** total records processed, run source (frozen/seed), `as_of` cutoff if one was applied — omit entirely if not relevant, don't show an empty "cutoff: none" card.

### 3.3 Exceptions — the honest exception list

**Purpose:** show, in detail, exactly what the system refused to resolve and why. This screen carries as much visual weight as Overview — it is not a secondary or "error log" screen.

**Reasoning trace data (backend note):** showing the near-duplicate candidate an `adversarial_trap` record considered and rejected requires structured data the exception-list API doesn't currently return (today it returns only `order_id`/`utr`/`status`/`confidence_note`, a single free-text sentence). This needs a small, explicitly-scoped addition: expose `evidence_tool_calls` and the candidate order_id(s) already present in `AgentResolution`/cached `debug_info`, as a new field on the exception-list response. No new agent logic or computation — purely surfacing data that already exists. Note this addition in `docs/plan.md`'s Layer 6 section as a disclosed small addendum, the same way the `as_of` fix was documented, rather than letting it live only as an undocumented UI-driven change.

**Layout:**

1. Header line: count + one sentence of framing ("These 17 records could not be confidently matched or explained. Each is preserved as a suspense entry in the ledger rather than a guess.")
2. **A table** (styled per Section 1.3's table treatment — zebra striping, distinct header, shadowed rounded container, row hover), one row per exception: Order/UTR reference · Category (as a small `COLOR_CAUTION`-tinted pill) · Reason (parsed, human-readable, from `confidence_note` — not the raw string) · Amount (monospace, right-aligned).
3. **Row expansion on click** reveals a distinctly-styled reasoning panel, not plain expander text: a left accent bar in the row's category color, slightly indented body text, clear visual separation from the table itself. For `adversarial_trap` rows, this panel shows the near-duplicate candidate the agent considered and rejected (using the new field above), labeled plainly: "Candidate considered and correctly rejected." This is the single strongest piece of evidence in the product — give it real visual room, not a cramped default expander.
4. **Empty state:** zero exceptions on a hypothetical run shows a calm one-line message ("No honest exceptions on this run"), never an empty table with just headers.

### 3.4 Ledger — trial balance

**Purpose:** the single most persuasive artifact for a finance-literate reviewer. Treat it with more formality than any other screen.

**Layout:**

1. A real table (same Section 1.3 treatment as Exceptions — shadowed container, zebra striping, distinct header — applied with restraint here, since this screen's correct feeling is closer to a real accounting statement than a product table): Account Code · Account Name · Type · Debit · Credit · Net.
2. Right-aligned monospace numbers throughout; a horizontal rule above the TOTAL row; TOTAL row bolded with a slightly heavier `COLOR_NEUTRAL_100` background.
3. **The TOTAL row's net column gets a visible checkmark badge** (`COLOR_SUCCESS`) next to "0.00" when balanced — the one moment in the whole UI where a success-green checkmark is earned. Don't dilute it by overusing the same color elsewhere.
4. Minimal additional decoration. The page overall should feel calm and authoritative, not sparse in a way that looks unfinished — the shadowed container and clean typography from Section 1 carry this screen; it doesn't need extra ornamentation on top of that.

### 3.5 Forecast — confirmed vs. projected

**Purpose:** demonstrate the ledger data is usable downstream, and visually distinguish fact from projection.

**Layout:**

1. A grouped bar chart, **full container width** (not a small fixed size) and visibly larger than a default Streamlit chart, centered within a max-width page container so it doesn't stretch awkwardly on a wide monitor. X-axis = date (7-day horizon), two series per date: Confirmed (`COLOR_SUCCESS`) and Projected (`COLOR_INFO`) — reusing the same two tokens already established for this exact distinction elsewhere in the product, not a separate palette (no purple, striped or otherwise). Projected bars carry a visible diagonal-hatch texture or reduced opacity in addition to color, so the distinction survives in grayscale too.
2. Projected bars show their ±5% confidence band as a thin error-bar-style whisker, labeled once in a legend ("± 5% settlement-day slip, illustrative") — not repeated per bar.
3. Below the chart: two summary cards, Total Confirmed and Total Projected, same card styling and hover-lift as Overview's metric cards (Section 1.5) — hovering one of these cards visually highlights (e.g., increased opacity/border emphasis on) its corresponding series in the chart above, the same linked-hover pattern as Overview's stacked bar. Values use `.money-figure` styling, `from_paise()`-derived.
4. The whole screen's content — chart, summary cards — sits in a centered container with consistent left/right margins, rather than left-aligned with empty space on one side.
5. If the loaded run has no in-flight amounts, show a one-line note: "This run is fully settled — nothing is currently projected. Trigger with a cutoff date to see an in-progress state." rather than an empty/flat chart with no explanation.

---

## 4. Interaction and state rules

- **Session state, not re-fetching.** Once a `batch_run_id` is selected, its data is fetched once and cached in `st.session_state`, not re-fetched on every widget interaction elsewhere on the page. Use `@st.cache_data` keyed on `(batch_run_id, as_of)` for each API-client call.
- **No silent staleness.** Triggering a new run while viewing an old one visibly replaces the old run's data, never blends it. The persistent run badge is the safeguard.
- **Errors are inline, never a raw traceback.** Any API error renders as a calm, `COLOR_CAUTION`-toned banner with a plain-English message and, where relevant, a suggestion — never a Streamlit exception traceback in a demo context.

---

## 5. Performance and responsiveness (Streamlit-specific)

1. **`@st.cache_data` on every API-client function**, keyed appropriately. The single highest-leverage fix — most Streamlit sluggishness is redundant recomputation on rerun, not network latency.
2. **`st.spinner()` around every network call longer than ~200ms.**
3. **Avoid `st.rerun()` calls except where truly necessary** (e.g., immediately after a successful trigger).
4. **Heavy computation** (chart bucketing, confidence-note parsing) happens once per data-fetch, cached alongside the raw response — not recomputed inline on every rerun.
5. **Page load target:** under 2 seconds to first meaningful content on a fresh `streamlit run`, assuming the API and Postgres are already warm.

---

## 6. What this spec deliberately does not include

No animations beyond Streamlit's built-in transitions and the subtle hover-lifts specified above, no custom JavaScript, no client-side routing library, no dark mode, no mobile-responsive breakpoints. If any of these seem worth adding later, propose and wait per `CLAUDE.md` §10 — don't add them silently while implementing this spec.

---

## 7. Build order for implementing this spec

1. **Backend addendum first, small and isolated:** the exception-list API field addition (Section 3.3) — implement and test this in isolation before touching any UI code, and document it in `docs/plan.md`'s Layer 6 section.
2. `tokens.py`/`theme.py` updates for any new tokens this spec introduces (accent border-left colors, hover-lift shadow values, hatch/opacity treatment for projected bars).
3. Layout restructure of `app.py`: the Run-then-four-views navigation (Section 2), `st.session_state`-based run persistence, and the Compare-mode toggle wired to the existing tested comparison logic — verified against existing tests before any visual polish.
4. Screen-by-screen visual implementation, in order: Run → Overview → Exceptions → Ledger → Forecast — each checked against its spec above before moving to the next.
5. Performance pass (Section 5) last, once the visual structure is settled.
