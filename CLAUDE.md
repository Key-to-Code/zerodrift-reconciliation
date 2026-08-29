# CLAUDE.md — Operating rules for this repository

You are building an AI-assisted financial reconciliation engine for Razorpay Buildathon
Track 04. Read `docs/context.md` (the "why") and `docs/plan.md` (the "how") in full before
writing any code. This file overrides your defaults and applies to every session, every
layer, without exception.

Time is not a constraint on this build. Build every layer in `docs/plan.md`, in order,
to its full acceptance criteria. Do not shrink, skip, or downgrade a layer to save time.
If something is genuinely ambiguous, stop and ask rather than guessing and moving on.

---

## 1. Integrity rules — not style preferences, do not negotiate these under any pressure

- **Never hardcode a test to pass.** A failing test means the code is wrong. Editing the
  assertion instead of the code is falsifying a result.
- **Never write a number into README.md, evaluate.py output, or any doc unless that exact
  number was produced by actually running the actual code against the actual dataset.**
  If the real number disappoints, report it and explain why — that is more credible than
  a fabricated one, not less.
- **Never fabricate, mock, or pre-bake a successful agent response.** An agent failure is
  real information. Surface it, don't paper over it.
- **Never invent sample data that isn't explicitly labelled synthetic.**
- **If something isn't working, stop and report it plainly** rather than producing an
  artifact that looks finished but silently isn't.

## 2. Architecture claims

This is a **modular monolith**: one Python process, one Streamlit process, one Postgres
instance, in-process calls, single ACID transactions. Never describe it as distributed,
as microservices, or as having cross-service failure handling — not in code, comments,
docstrings, README, or pitch material. If you find such language anywhere, flag it and
remove it.

The only Razorpay-internal claim permitted anywhere is the verified fact that Agent
Studio runs on Anthropic's Claude Agent SDK. Assert nothing else about their internal
stack, verified or not.

## 3. Money

- `Decimal` at the Pydantic boundary, parsed from `str`. Never from a `float` literal.
  A `field_validator` rejects float input outright.
- Integer **paise** inside Polars. Never `Float64`, never an inferred schema on a money
  column.
- `NUMERIC` in Postgres.
- Exactly two conversion functions, `to_paise()` / `from_paise()`, living in one module.
  They have a round-trip test over the full frozen dataset, zero drift. No ad-hoc
  conversions anywhere else in the codebase.
- If a `float` ever appears on a path that touches money, that is a bug, not a shortcut,
  full stop.

## 4. Domain constants — verified, do not re-derive or guess these

- GST on MDR: **18%**.
- TDS under Section 194-O: **0.1%**, reduced from 1% effective 01-Oct-2024. Never use 1%
  anywhere in the generator, the tax-rule tool, or documentation.
- **UPI P2M MDR is nil by regulation.** Therefore MDR = 0 and GST-on-MDR = 0 on every UPI
  record. The generator must never place a `missing_tax_line` anomaly on a UPI
  transaction — a UPI record with a nonzero tax line to omit is an impossible
  transaction, and it is the kind of thing a finance-literate reviewer catches in seconds.
- GST-on-MDR is Input Tax Credit → an **asset**. TDS withheld is an advance tax on the
  merchant's behalf → an **asset**. Neither is ever a liability. All accounts are
  classified from the **merchant's** books, not the gateway's or the bank's.
- State the 194-O modeling assumption explicitly in the README (see Layer 9): this
  project models the merchant as an e-commerce participant settling through an operator
  who deducts under 194-O. It does not resolve the broader debate about whether a pure
  payment aggregator is in scope of that section — say so in one sentence, don't dodge it.
- Settlement windows: UPI T+1, cards T+2, international T+3, counted in **business days**
  against an IST calendar that excludes weekends and a small hardcoded list of Indian
  bank holidays. A ±2 calendar-day window that includes a weekend will falsely flag a
  clean transaction — this is a real, not theoretical, failure mode. See Layer 1/2.

## 5. Reproducibility — the project's central claim, treat it literally

- The **deterministic path** (generator, fast path, ledger posting, paise round-trip,
  trial balance) must be byte-reproducible for a given seed. This is asserted by tests,
  not just claimed.
- The **agent path is not deterministic** and must never be described or reported as if
  it were. `evaluate.py` runs the agent path 3 times per batch and reports
  min/median/max plus variance, not a single number.
- Every agent invocation (inputs, tool calls, final resolution) is logged to
  `data/agent_runs/<seed>_<run_index>.jsonl`. `evaluate.py --replay <path>` re-scores
  from a cached run without calling the model again — this is the offline demo fallback
  if the venue network or the API is down.
- The README's scorecard is split into a deterministic block (single exact numbers) and
  an agent block (ranges across 3 runs). Never merge these into one misleadingly precise
  figure.

## 6. Money splitting across a UTR batch — largest-remainder allocation

Splitting an integer-paise lump-sum bank credit across N orders in a UTR batch will
almost always leave 1 to N-1 paise unallocated to naive proportional division. This is
not an edge case — it happens on the majority of multi-order batches. Handle it with the
largest-remainder method: compute each order's exact share, take the floor, then
distribute the leftover paise one each to the orders with the largest fractional
remainder, until the total matches the bank credit exactly. Post any genuinely
unexplained residual (never more than ± batch_size paise) to a new **`ROUNDING_DIFFERENCE`**
expense account, added to the chart of accounts in Layer 3 — asserted in a test, never
silently swallowed or force-balanced by fudging one order's amount.

## 7. Do not add these, ever, under any framing

Next.js, pgvector, full OpenTelemetry tracing, Kafka, Spark, any streaming
infrastructure, any message broker, any microservice split. Each was considered and
deliberately cut during planning. Prefer the version of a decision that can be fully
defended in two sentences over the version that sounds more advanced.

## 8. Build protocol — follow this exactly, every layer

1. Build in **layer order**, per `docs/plan.md`. Do not start layer N+1, and do not "get
   a head start" on a later layer's code, until every acceptance checkbox in layer N is
   demonstrably met.
2. For each layer: **write the tests first**, derived directly from that layer's
   acceptance criteria. Show me the test list before implementing and wait for approval.
3. Implement until the tests pass honestly — no adjusting the test to match broken output.
4. Run the **whole** suite, `pytest tests/ -v`, not just the new file's tests, and paste
   the real output.
5. Report which acceptance criteria pass and which do not, each with the specific test
   that proves it. Never claim a criterion is met with no test behind it.
6. `git commit` with the layer number in the message. Minimum one commit per layer.

## 9. Frozen artifacts

`data/challenge_batch_100/` is generated exactly once, committed, and never regenerated.
If any later change would alter what it contains, stop and ask before touching it.
`ground_truth.json` is the single most important file in the repository — every claim the
project makes is only as credible as this file, including the per-record
`expected_root_cause_code` and `expected_delta_paise` fields used to score agent
diagnoses in Layer 4/8.

## 10. Scope discipline

If you believe something outside `docs/plan.md` should be added — even a genuine
improvement — **propose it and wait for approval**. Do not add it silently mid-layer.
Distinguish clearly, out loud, between a real bug fix (proceed once flagged) and new
scope disguised as a fix (flag explicitly as optional, do not add without approval).
