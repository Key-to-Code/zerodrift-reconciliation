# ZeroDrift — Complete Build Plan (v2, audit-integrated)
### Razorpay Buildathon Track 04 · For execution by Claude Code

**Read this whole file, and `CLAUDE.md`, before writing any code.** Build strictly in
layer order. Each layer has explicit acceptance criteria — do not start the next layer
until the current one's criteria pass, with a test behind every checkbox. This document
supersedes all earlier drafts; where it differs from an earlier plan, this version wins.

**Changes from the previous draft, and why**, so nothing here reads as unexplained:
1. Reproducibility claim split into a deterministic block and a 3-run agent-variance
   block (was: one unreproducible number presented as exact).
2. Agent resolutions are now a structured, gradeable schema (`root_cause_code` +
   `quantified_delta_paise`), diffed against ground truth (was: "did it not crash" only).
3. UTR batch cash splitting uses largest-remainder allocation with a `ROUNDING_DIFFERENCE`
   account (was: unhandled paise residue that would fail the balance trigger on real data).
4. TDS 194-O rate corrected to 0.1% (was implicitly unspecified / risk of using the old
   1% rate); UPI records are now constrained to nil MDR/GST (was: no such constraint,
   allowing an impossible transaction to be generated).
5. Date-window matching now runs on a business-day IST calendar (was: raw calendar days,
   which breaks on any window crossing a weekend).
6. Two new anomaly categories — `short_settlement` and `duplicate_credit` — added to the
   generator (was: orphan bank credit was the only "no clean counterpart" case modeled).
7. Demo reset changed from destructive `TRUNCATE` to a `batch_run_id`-scoped design (was:
   wiping the frozen batch's ledger rows to load a live seed, losing the comparison).
8. Layers 5 (forecaster) and 6 (API) restored to full scope — no time-boxing, no cuts.

**Architecture type:** modular monolith. One Python process, one Streamlit process, one
PostgreSQL instance. No microservices, no message broker, no distributed transactions.
See `CLAUDE.md` §2 for the exact framing to use everywhere this comes up.

---

## Layer 0 — Environment setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install polars pydantic pydantic-settings rapidfuzz pytest rich python-dateutil
```

Add SQLAlchemy, LangGraph, anthropic, FastAPI, Streamlit, and Docker tooling when their
respective layer starts — not before.

### Project structure

```
razorpay-ai-finance-controller/
├── CLAUDE.md
├── docs/
│   ├── context.md
│   └── plan.md
├── venv/
├── src/
│   ├── __init__.py
│   ├── common/
│   │   ├── __init__.py
│   │   ├── money.py              # to_paise / from_paise (Layer 1/2 boundary)
│   │   └── calendar.py           # IST business-day calendar (Layer 1/2)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── models.py             # Pydantic schemas (Layer 1)
│   │   └── generator.py          # synthetic dataset generator (Layer 1)
│   ├── matching/
│   │   ├── __init__.py
│   │   ├── schema.py             # Polars schemas (Layer 2)
│   │   └── fast_path.py          # 3-hop matching cascade (Layer 2)
│   ├── ledger/
│   │   ├── __init__.py
│   │   ├── db_schema.sql         # Postgres DDL (Layer 3)
│   │   ├── models.py             # SQLAlchemy models (Layer 3)
│   │   ├── allocation.py         # largest-remainder UTR split (Layer 3)
│   │   └── journal.py            # posting logic + gatekeeper (Layer 3)
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── tools.py              # read-only diagnostic tools (Layer 4)
│   │   ├── resolution.py         # AgentResolution schema (Layer 4)
│   │   └── graph.py              # LangGraph bounded loop (Layer 4)
│   ├── forecast/
│   │   ├── __init__.py
│   │   └── cashflow.py           # rolling forecaster (Layer 5)
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py               # FastAPI entrypoint (Layer 6)
│   └── dashboard/
│       └── app.py                # Streamlit (Layer 7)
├── tests/
│   ├── test_common.py
│   ├── test_generator.py
│   ├── test_fast_path.py
│   ├── test_ledger.py
│   └── test_agent.py
├── data/
│   ├── challenge_batch_100/      # frozen, committed, never regenerated after Day 1
│   └── agent_runs/                # cached agent invocations, for --replay
├── evaluate.py                   # Layer 8
├── docker-compose.yml            # Layer 9 (Postgres only)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Layer 1 — Data: Pydantic schemas + synthetic generator

### 1.1 `src/common/calendar.py` — build this first, Layer 1 depends on it

An IST business-day calendar: excludes Saturday/Sunday and a small hardcoded list of
Indian bank holidays covering the demo period (Republic Day, Independence Day, Gandhi
Jayanti, Diwali — pick the actual dates for the relevant year; a short fixed list is
fine, this does not need to be exhaustive or externally sourced). Expose:

```python
def add_business_days(d: date, n: int) -> date: ...
def business_days_between(d1: date, d2: date) -> int: ...
def is_business_day(d: date) -> bool: ...
```

This is shared by the generator's `cutoff_drift` timing and by Layer 2's ±N-day window
matching. A ±2 **calendar**-day window silently fails any settlement whose window
crosses a weekend — this must be business days everywhere a date window is enforced.

### 1.2 `src/common/money.py`

`to_paise(Decimal) -> int` and `from_paise(int) -> Decimal`. These are the only two
functions in the codebase allowed to convert between `Decimal` and integer paise. Every
other module imports these rather than reimplementing the conversion.

### 1.3 `src/data/models.py`

Three record types. All monetary fields `Decimal`, parsed from string — never a `float`
literal.

- **InternalOrder**: `order_id: str`, `gross_amount: Decimal`, `customer_id: str`,
  `payment_method: Literal["upi","credit_card","debit_card","netbanking","amex"]`,
  `timestamp: datetime`
- **GatewaySettlement**: `payment_id: str`, `order_id: str`, `gross_amount: Decimal`,
  `mdr: Decimal`, `gst_on_mdr: Decimal`, `tds_194o: Decimal`, `net_amount: Decimal`,
  `utr: str`, `settlement_date: date`
- **BankStatementLine**: `utr: str`, `credited_amount: Decimal`, `value_date: date`,
  `narration: str`

Validators:
- All monetary fields non-negative.
- `net_amount == gross_amount - mdr - gst_on_mdr - tds_194o` exactly, zero tolerance.
- Reject construction from `float` for monetary fields — raise in a `field_validator`.
- **FIX (domain):** if `payment_method == "upi"`, `mdr` and `gst_on_mdr` must both be
  exactly zero. Enforce this as a model validator, not just a generator convention — a
  test should attempt to construct a UPI settlement with nonzero MDR and confirm it is
  rejected.
- **Timezone pinning on `InternalOrder.timestamp`, required.** IST (UTC+5:30), pinned
  explicitly:
```python
from datetime import datetime
import zoneinfo

IST = zoneinfo.ZoneInfo("Asia/Kolkata")

@field_validator("timestamp")
@classmethod
def ensure_ist(cls, v: datetime) -> datetime:
    if v.tzinfo is None:
        return v.replace(tzinfo=IST)
    return v.astimezone(IST)
```
`GatewaySettlement.settlement_date` and `BankStatementLine.value_date` stay plain `date`.

### 1.4 `src/data/generator.py`

CLI: `python -m src.data.generator --records 100 --seed 42 --out data/batch_100`

Deterministic — same seed produces byte-identical output every run. Outputs 4 files:
`internal_orders.json`, `gateway_settlement.json`, `bank_statement.json`,
`ground_truth.json`.

`ground_truth.json` entry per order:
```json
{
  "order_id": "ORD1042",
  "category": "fee_drift",
  "expected_resolution": "agent_resolved",
  "expected_root_cause_code": "AMEX_SURCHARGE",
  "expected_delta_paise": 1180,
  "notes": "Amex 2.5% surcharge not in standard MDR table"
}
```
`expected_resolution` is one of `fast_path`, `agent_resolved`, `honest_exception`.
`expected_root_cause_code` and `expected_delta_paise` are populated for every
`agent_resolved` record — see Layer 4's `AgentResolution` schema; these two fields are
what makes "the agent resolved it correctly" a checkable claim instead of an assertion.
Leave both `null` for `fast_path` and `honest_exception` rows.

**FIX (domain constants):** TDS on `GatewaySettlement` uses **0.1%**, not 1%. GST on MDR
is 18%. These live as named constants in the generator, imported wherever else they're
needed — never re-typed as a magic number in a second place.

Category mix for a 100-record batch (scale proportionally for other sizes; the trap
category is never zero):

| Category | Count | Logic |
|---|---|---|
| clean_match | 56 | order→payment→UTR→bank all consistent, standard MDR by rail |
| utr_batch | 10 | 2–4 orders grouped under one UTR; bank credit = sum of net amounts, split via largest-remainder allocation (Layer 3) |
| cutoff_drift | 5 | order timestamp 23:45–23:59 IST; settlement pushed T+1/T+2 in **business days** |
| fee_drift | 7 | Amex/corporate card, actual MDR different from standard rate table |
| missing_tax_line | 5 | `gst_on_mdr` or `tds_194o` omitted/zeroed in settlement feed — **never on a UPI record** (see 1.3 validator) |
| orphan | 8 | bank credit with UTR matching no settlement record at all |
| adversarial_trap | 5 | two unrelated orders, near-identical amounts, overlapping dates, no real link |
| **short_settlement** | **2** | **NEW.** Settlement record exists with a valid UTR, but no bank credit for that UTR ever appears — funds held or a rail failure. Distinct from `orphan` (which is a bank credit with nothing behind it); this is the mirror case: a settlement with nothing behind it. Expected resolution: `honest_exception`. |
| **duplicate_credit** | **2** | **NEW.** The same UTR is credited twice in the bank statement (a real, if rare, production failure mode). The cardinality guardrail in Layer 2 must catch this — two candidates passing all thresholds for one settlement record — and route to the discrepancy queue rather than auto-matching either one. Expected resolution: `honest_exception` (a human confirms which credit, if either, is a genuine duplicate refundable to the bank). |

Total: 56+10+5+7+5+8+5+2+2 = 100.

Bank `narration` field must be realistically messy (e.g. "NEFT-UTR8827301-SETTLE",
"IMPS/RZPY/882730/xx") — this is what rapidfuzz gets tested against in Layer 2. Do not
generate clean narrations.

**Anti-self-grading requirement.** Category values must be randomized within realistic
ranges, not fixed constants — a fixed `fee_drift` delta lets the agent's tool lookup
trivially special-case it, making the test circular. Randomize magnitudes (e.g. Amex
surcharge as 2.0%–3.0%, not a fixed rupee amount), which specific orders get which
treatment, and timing offsets within each category's window.

**Unseen challenge batch.** In addition to the frozen `challenge_batch_100`, the
generator must run live with a seed nobody has seen (`--seed <judge-provided-seed>`),
producing a fresh batch with the same category distribution but different values. This
is the strongest available counter to "your results are cherry-picked" — offer it in
the live demo.

**Refund/chargeback category — now in scope, not a stretch goal, since time is not
constrained.** Add a `refund_clawback` category (3 records). Under Indian gateway
mechanics, a customer refund debits `REVENUE_GROSS` and credits `AR_GATEWAY_CLEARING`
without reversing the previously-paid `MDR_EXPENSE` — the merchant does not recover the
processing fee on a refunded transaction. Document this explicitly in `journal.py`
(Layer 3) and give it its own `expected_resolution` and, where it requires agent
judgment, its own `root_cause_code` (e.g. `REFUND_NO_MDR_REVERSAL`). If you add this,
recompute the category-mix table above to include it (adjust `clean_match` down to
absorb the extra 3, keeping the total at 100).

**Layer 1 acceptance criteria (all must pass before Layer 2 starts):**
- [ ] Generator runs, produces all 4 files
- [ ] Same seed produces byte-identical output across two runs
- [ ] `ground_truth.json` has 100 entries, category counts match spec (±1)
- [ ] Every `agent_resolved` entry has a non-null `expected_root_cause_code` and
      `expected_delta_paise`; every `fast_path`/`honest_exception` entry has both null
- [ ] No UPI record in `gateway_settlement.json` has nonzero `mdr` or `gst_on_mdr`
- [ ] TDS on every settlement record is computed at 0.1%, not 1% — asserted directly
- [ ] `short_settlement` and `duplicate_credit` categories each present with the
      counts specified, and are distinguishable from `orphan` in the ground truth
- [ ] 5 or more adversarial trap records present and logged
- [ ] `pytest tests/test_generator.py` passes: no floats anywhere in output, `net_amount`
      invariant holds on every settlement record, orphans and short_settlements have
      zero valid match anywhere in the batch
- [ ] `pytest tests/test_common.py` passes: `to_paise`/`from_paise` round-trip exactly,
      and the business-day calendar correctly skips weekends and the hardcoded holidays
- [ ] `data/challenge_batch_100/` committed to git and never regenerated after this point

---

## Layer 1 Addendum — clarifications resolved before implementation (2026-08-29)

Six ambiguities were raised against the spec above and resolved as follows. These
resolutions are binding for the generator and model implementation; if a later layer
needs to revisit one, flag it explicitly rather than silently diverging.

**A1 — `missing_tax_line` vs. the `net_amount` exact-invariant validator.**
The model validator (`net_amount == gross - mdr - gst - tds`, zero tolerance) cannot be
relaxed, so a "missing tax line" cannot mean "the line is zeroed but net_amount still
reflects the true deduction" — that record would fail to construct at all. Resolution:
the anomaly is modeled as a genuine settlement computation error — the omitted tax truly
was not deducted, so `net_amount` is recomputed consistent with the zeroed line (the
merchant was overpaid by that tax amount). The record stays internally consistent with
the invariant; the discrepancy is that it's inconsistent with the *expected* tax rules
(GST 18% / TDS 0.1%), which is what the fast path / agent must catch by recomputing the
expected values and comparing.

**A2 — `refund_clawback` record shape.** No new top-level schema. Add an optional
`refund_amount: Decimal | None = None` field to `InternalOrder` (same `Decimal`-from-str
discipline as other money fields; when set, must be strictly positive and less than
`gross_amount`). A record with `refund_amount` set is the `refund_clawback` category. The
corresponding `GatewaySettlement` is unaffected by the refund (MDR/GST/TDS computed on
the original gross amount, never reversed) — this is what makes the category meaningful.
Ledger-level handling of this field is Layer 3's concern; Layer 1 only needs to generate
and label the data correctly.

**A3 — `INTL_MARKUP` / "corporate card" fee drift needs an international marker.**
Add `is_international: bool = False` to `GatewaySettlement`. `fee_drift` records may set
this on `amex` or `credit_card` rails to justify an additional markup on top of the
standard rate table, consistent with the `AMEX_SURCHARGE` / `INTL_MARKUP` root-cause
codes in Layer 4. No new payment methods are added — `payment_method` stays exactly the
five-value literal from §1.3.

**A4 — Ground-truth keying for order-less anomalies.** `orphan` records (a bank credit
with no settlement or order behind it at all) get a synthetic identifier of the form
`UNMATCHED_BANK_<utr>` in place of `order_id` in their `ground_truth.json` entry. This
prefix is reserved — no real `InternalOrder.order_id` may ever start with
`UNMATCHED_BANK_`, enforced as a generator-level test.

**A5 — Dataset date range and holiday list, fixed in code (not wall-clock-derived).**
Because determinism (criterion 2) requires byte-identical output for a given seed
regardless of *when* the generator is run, all dates are anchored to a fixed window
hardcoded in `src/data/generator.py`, never to `date.today()`: **2025-01-06 through
2025-02-02** (four weeks, includes at least one full weekend-crossing window for
`cutoff_drift`). `src/common/calendar.py`'s hardcoded Indian bank holiday list covers
2025: Republic Day (2025-01-26, falls on a Sunday), Independence Day (2025-08-15),
Gandhi Jayanti (2025-10-02), Diwali (2025-10-20). The list exists mainly to prove the
calendar logic correctly excludes a holiday that *isn't* already a weekend — Independence
Day 2025-08-15 is a Friday — even though that date falls outside the generator's own
batch window.

**A6 — `adversarial_trap`'s `expected_resolution`.** Confirmed as `honest_exception` for
all 5 records, consistent with `orphan`, `short_settlement`, and `duplicate_credit`.

**Category mix, recomputed with `refund_clawback` included (A2), total still 100:**

| Category | Count |
|---|---|
| clean_match | 53 |
| utr_batch | 10 |
| cutoff_drift | 5 |
| fee_drift | 7 |
| missing_tax_line | 5 |
| orphan | 8 |
| adversarial_trap | 5 |
| short_settlement | 2 |
| duplicate_credit | 2 |
| refund_clawback | 3 |

---

## Layer 2 — Fast path: Polars matching cascade

### 2.1 `src/matching/schema.py`

Explicit Polars schemas — never let Polars infer money columns, or it silently coerces
to `Float64`. Money stored as integer paise, converted at the boundary via
`src/common/money.py` only.

```python
orders_schema = {
    "order_id": pl.Utf8, "gross_amount_paise": pl.Int64,
    "customer_id": pl.Utf8, "payment_method": pl.Categorical,
    "timestamp": pl.Datetime("us"),
}
settlement_schema = {
    "payment_id": pl.Utf8, "order_id": pl.Utf8, "gross_amount_paise": pl.Int64,
    "mdr_paise": pl.Int64, "gst_on_mdr_paise": pl.Int64, "tds_paise": pl.Int64,
    "net_amount_paise": pl.Int64, "utr": pl.Utf8, "settlement_date": pl.Date,
}
bank_schema = {
    "utr": pl.Utf8, "credited_amount_paise": pl.Int64,
    "value_date": pl.Date, "narration": pl.Utf8,
}
```
`order_id` and `utr` stay `pl.Utf8`, exact string equality, no premature categorical
coercion.

### 2.2 `src/matching/fast_path.py` — the 3-hop cascade

1. **order to payment**: exact key join on `order_id`.
2. **payment to UTR**: group settlements by `utr`, aggregate `net_amount_paise` as an
   exact integer sum. If the sum doesn't reconcile against the bank credit, fall through
   to the agent rather than force it.
3. **UTR to bank**: two-phase matching.
   - **Phase 1 — regex identifier extraction:** regex for known alphanumeric tokens
     (`UTR[A-Z0-9]+`, `ORD[0-9]+`) against the narration. Exact extracted token match
     wins immediately, no fuzzy scoring needed.
   - **Phase 2 — fallback only if Phase 1 finds no exact token:** `rapidfuzz.fuzz.
     partial_ratio` (not `token_sort_ratio` — truncation tolerant), threshold ≥85, plus
     amount equality **and a ±2 business-day value-date window** (via
     `src/common/calendar.py` — **FIX**: this was a raw calendar-day window before,
     which breaks on any window crossing a weekend) as hard constraints. Narration
     similarity alone, at either phase, is never sufficient by itself.

**Cardinality guardrail — required.** If either phase returns more than one candidate
passing all thresholds, do NOT auto-match the first hit. Exactly 1 candidate → auto-match
via fast path. More than 1 → route to the Layer 4 discrepancy queue. This is also the
mechanism that correctly catches the new `duplicate_credit` category: two bank lines on
the same UTR both pass thresholds, cardinality > 1, routed to the queue rather than
silently taking the first (and possibly wrong) one.

Anything that doesn't clear all three hops cleanly — including `short_settlement` (no
bank credit ever appears for that UTR) — goes to the discrepancy queue for Layer 4.

**Layer 2 acceptance criteria:**
- [ ] `to_paise`/`from_paise` round-trip exactly on the full challenge batch, zero drift
- [ ] Fast path resolves roughly 60–70% of the batch (`clean_match` plus `utr_batch`)
      with zero false positives against `ground_truth.json`
- [ ] Adversarial trap records are NOT resolved by the fast path
- [ ] A test with two records sharing identical amount, date, and near-identical
      narration confirms both route to the discrepancy queue
- [ ] The `duplicate_credit` records are correctly caught by the cardinality guardrail,
      not auto-matched to either candidate
- [ ] A test confirms a window that crosses a weekend still matches correctly under the
      business-day calendar, and would have failed under a naive calendar-day window
      (write this as an explicit regression test, not just a passing case)
- [ ] `pytest tests/test_fast_path.py` passes

---

## Layer 3 — Ledger: PostgreSQL double-entry system

### 3.1 `src/ledger/db_schema.sql`

```sql
CREATE TYPE entry_status AS ENUM ('posted', 'reversed');
CREATE TYPE account_type AS ENUM ('asset', 'liability', 'revenue', 'expense', 'suspense');
CREATE TYPE match_status AS ENUM ('fast_path', 'agent_resolved', 'honest_exception');

CREATE TABLE accounts (
    account_id SERIAL PRIMARY KEY,
    account_code VARCHAR(50) UNIQUE NOT NULL,
    account_name VARCHAR(200) NOT NULL,
    account_type account_type NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE journal_entries (
    entry_id BIGSERIAL PRIMARY KEY,
    batch_run_id UUID NOT NULL,                      -- FIX: was missing; enables non-destructive demo resets
    idempotency_key VARCHAR(120) UNIQUE NOT NULL,
    reference_id VARCHAR(100) NOT NULL,
    description TEXT,
    status entry_status NOT NULL DEFAULT 'posted',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE journal_lines (
    line_id BIGSERIAL PRIMARY KEY,
    entry_id BIGINT NOT NULL REFERENCES journal_entries(entry_id),
    account_id INT NOT NULL REFERENCES accounts(account_id),
    direction CHAR(1) NOT NULL CHECK (direction IN ('D','C')),
    amount NUMERIC(14,2) NOT NULL CHECK (amount > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE reconciliation_matches (
    match_id BIGSERIAL PRIMARY KEY,
    batch_run_id UUID NOT NULL,
    order_id VARCHAR(100),
    payment_id VARCHAR(100),
    utr VARCHAR(100),
    status match_status NOT NULL,
    confidence_note TEXT,
    journal_entry_id BIGINT REFERENCES journal_entries(entry_id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (batch_run_id, order_id)                  -- FIX: was missing; prevented double-inserting a match
);

CREATE INDEX idx_journal_lines_entry ON journal_lines(entry_id);
CREATE INDEX idx_journal_entries_reference ON journal_entries(reference_id);
CREATE INDEX idx_journal_entries_run ON journal_entries(batch_run_id);
CREATE INDEX idx_recon_order ON reconciliation_matches(order_id);
CREATE INDEX idx_recon_utr ON reconciliation_matches(utr);
CREATE INDEX idx_recon_run ON reconciliation_matches(batch_run_id);

CREATE OR REPLACE FUNCTION check_entry_balances() RETURNS TRIGGER AS $$
DECLARE
    target_entry_id BIGINT;
    debit_total NUMERIC(14,2);
    credit_total NUMERIC(14,2);
BEGIN
    target_entry_id := COALESCE(NEW.entry_id, OLD.entry_id);
    SELECT COALESCE(SUM(amount) FILTER (WHERE direction='D'), 0),
           COALESCE(SUM(amount) FILTER (WHERE direction='C'), 0)
    INTO debit_total, credit_total
    FROM journal_lines WHERE entry_id = target_entry_id;

    IF debit_total <= 0 OR credit_total <= 0 THEN
        RAISE EXCEPTION 'Journal entry % has non-positive balance totals (Debits: %, Credits: %)',
            target_entry_id, debit_total, credit_total;
    END IF;
    IF debit_total != credit_total THEN
        RAISE EXCEPTION 'Unbalanced journal entry %: debits % != credits %',
            target_entry_id, debit_total, credit_total;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE CONSTRAINT TRIGGER trg_check_balance
    AFTER INSERT OR UPDATE OR DELETE ON journal_lines
    DEFERRABLE INITIALLY DEFERRED
    FOR EACH ROW EXECUTE FUNCTION check_entry_balances();

-- Accounts are classified from the MERCHANT's books, not Razorpay's.
-- GST paid on the MDR fee is Input Tax Credit (an asset) — not a liability.
-- TDS withheld under Section 194-O (0.1%) is an advance tax asset — not a liability.
INSERT INTO accounts (account_code, account_name, account_type) VALUES
    ('CASH', 'Cash / Bank (Nodal Escrow)', 'asset'),
    ('CASH_IN_TRANSIT_UTR', 'Cash In Transit — UTR Batch Clearing', 'asset'),
    ('AR_GATEWAY_CLEARING', 'Gateway Clearing Receivable', 'asset'),
    ('REVENUE_GROSS', 'Merchant Gross Sales', 'revenue'),
    ('MDR_EXPENSE', 'Payment Processing Fee (MDR)', 'expense'),
    ('GST_ITC_RECEIVABLE', 'GST Input Tax Credit on MDR', 'asset'),
    ('TDS_194O_CREDIT', 'TDS Withheld Asset (Section 194-O, 0.1%)', 'asset'),
    ('ROUNDING_DIFFERENCE', 'UTR Batch Allocation Rounding', 'expense'),   -- NEW, FIX
    ('SUSPENSE_UNRESOLVED', 'Unresolved Reconciliation Suspense', 'suspense');
```

**Residual gap to know about, not solve with more triggers:** a `journal_entries` row
inserted with zero `journal_lines` never fires the lines trigger. Close this at the
application layer instead: `journal.py`'s posting function always inserts an entry and
its lines in one function, one transaction. Add a test-level invariant asserting every
`journal_entries` row has at least 2 matching, balancing `journal_lines` rows.

**Two-stage transaction lifecycle — required.**

Stage 1 — `POST_ORDER_CAPTURE`, at order capture (T0):
```
Debit  AR_GATEWAY_CLEARING   (gross amount)
Credit REVENUE_GROSS          (gross amount)
```
Stage 2 — `POST_SETTLEMENT_CLEARING`, at settlement:
```
Debit  CASH_IN_TRANSIT_UTR   (net share of the settlement)
Debit  MDR_EXPENSE            (fee deducted)
Debit  GST_ITC_RECEIVABLE     (GST paid on the fee)
Debit  TDS_194O_CREDIT        (0.1% tax withheld on gross)
Credit AR_GATEWAY_CLEARING    (gross order amount — clears Stage 1 to zero)
```
UTR lump-sum bank credit — posts once per UTR:
```
Debit  CASH                   (full lump-sum bank credit for the UTR)
Credit CASH_IN_TRANSIT_UTR    (full lump-sum bank credit for the UTR)
```
For a `clean_match` (non-batched) order, Stage 2 and the UTR posting collapse into one
entry crediting `AR_GATEWAY_CLEARING` and debiting `CASH` directly.

### 3.2 `src/ledger/allocation.py` — NEW, largest-remainder UTR split

**FIX (audit item 4):** splitting an integer-paise lump sum across N orders by simple
proportional division almost always leaves 1 to N−1 paise unallocated — the DB trigger
will reject the resulting entries as unbalanced. Use largest-remainder allocation:

```python
def allocate_utr_batch(total_paise: int, shares: list[tuple[str, int]]) -> dict[str, int]:
    """
    shares: list of (order_id, exact_net_share_paise) BEFORE rounding.
    Returns {order_id: allocated_paise} summing exactly to total_paise.
    Any unexplained residual beyond simple floor/remainder distribution (which should
    not occur if `shares` sums correctly pre-rounding) is posted separately to
    ROUNDING_DIFFERENCE by the caller, never silently absorbed into one order's amount.
    """
```
Cap any residual posted to `ROUNDING_DIFFERENCE` at `± len(shares)` paise per batch, and
assert this cap in a test — a larger residual means an upstream allocation bug, not
rounding, and must not be swept into this account.

### 3.3 `src/ledger/journal.py` — posting logic and gatekeeper

Before any DB write: revalidate the entry through a Pydantic model checking
`sum(debit lines) == sum(credit lines)` in application code — the DB trigger is the hard
backstop, the Pydantic check is the fast-fail with a clean error message.

For `honest_exception` records: post a balancing entry against `SUSPENSE_UNRESOLVED`
rather than leaving it unposted, so the books stay balanced.

**Idempotency key scheme — now scoped by `batch_run_id` (FIX, replaces the earlier
truncate-based demo reset):**
```python
idempotency_key = f"RUN:{batch_run_id}:ORDER:{order_id}:SETTLE"
idempotency_key = f"RUN:{batch_run_id}:ORDER:{order_id}:UTR:{utr}:SETTLE"
idempotency_key = f"RUN:{batch_run_id}:BANK_TXN:{bank_txn_id}:SUSPENSE"
```
Every `journal_entries` and `reconciliation_matches` row carries `batch_run_id`. This
means the frozen `challenge_batch_100` run and a judge's live unseen-seed run coexist in
the same database, queryable and comparable side by side, instead of one destroying the
other. **Drop the earlier `TRUNCATE`-based `reset_database_for_demo()` design** — it is
no longer needed as the primary mechanism now that runs are scoped, though keeping a
`wipe_all_runs()` utility for local dev convenience is fine as long as it is never wired
to anything the demo path depends on.

### 3.4 Trial balance — NEW, for both README credibility and Layer 7

```python
def trial_balance(batch_run_id: UUID) -> pl.DataFrame:
    """Every account, its closing debit/credit balance for this run, and a final
    total row that must sum to zero. This is the single most persuasive artifact
    to show a finance-literate evaluator — print it in full in the README and in
    the Streamlit dashboard, not just a PASS/FAIL boolean."""
```

**Layer 3 acceptance criteria:**
- [ ] Deliberately unbalanced entry gets rejected by the Pydantic pre-check and, if
      bypassed in a test, by the DB trigger
- [ ] Deleting a line from a previously-balanced entry is caught by the trigger
- [ ] Every `journal_entries` row has ≥2 matching, balanced `journal_lines` rows —
      asserted as a test invariant
- [ ] Posting the same entry twice with the same idempotency key does not duplicate it
- [ ] GST and TDS post to `GST_ITC_RECEIVABLE` / `TDS_194O_CREDIT` (assets), never to
      a liability account
- [ ] `REVENUE_GROSS` is actually posted to at Stage 1 — confirm it's not unused
- [ ] `CASH_IN_TRANSIT_UTR` nets to zero once a full UTR batch's orders have all posted
      their Stage 2 entries
- [ ] Largest-remainder allocation on every `utr_batch` record in the frozen dataset
      sums exactly to the bank credit, zero unexplained residual beyond the asserted cap
- [ ] Running the frozen batch and a second, differently-seeded batch in the same
      database (different `batch_run_id`s) produces two independently correct trial
      balances with no cross-contamination
- [ ] `trial_balance()` sums to exactly zero for a fully-settled run
- [ ] `pytest tests/test_ledger.py` passes
- [ ] Fast-path results from Layer 2 post correctly into the ledger

---

## Layer 4 — Agent: bounded diagnostic loop

### 4.1 `src/agent/resolution.py` — NEW, the structured resolution schema

**FIX (audit item 3):** without this, "the agent resolved it" only means "the agent
produced something," not "the agent was right." Make the diagnosis gradeable:

```python
from typing import Literal
from pydantic import BaseModel

RootCauseCode = Literal[
    "AMEX_SURCHARGE", "INTL_MARKUP", "MISSING_GST", "MISSING_TDS",
    "CUTOFF_T1", "CUTOFF_T2", "BATCH_LEVEL_FEE",
    "REFUND_NO_MDR_REVERSAL",  # if the refund_clawback category from Layer 1 is included
    "UNRESOLVED",
]

class AgentResolution(BaseModel):
    root_cause_code: RootCauseCode
    quantified_delta_paise: int
    evidence_tool_calls: list[str]
    confidence_note: str
```
`evaluate.py` (Layer 8) diffs `root_cause_code` and `quantified_delta_paise` against
`ground_truth.json`'s `expected_root_cause_code` / `expected_delta_paise` fields. A
resolution that avoids "honest_exception" but gets the wrong root cause or a wrong
delta is scored as **incorrect**, not merely "resolved" — this is what makes the
diagnosis claim real rather than cosmetic.

### 4.2 `src/agent/tools.py` — read-only tools only, no ledger write access

Tool outputs require reasoning, not a scalar return:
```json
{
  "merchant_id": "m_1029",
  "effective_date": "2026-01-01",
  "pricing_model": "TIERED_BLENDED",
  "schedules": [
    {"rail": "amex_corporate", "base_rate": "0.025", "international_markup": "0.005", "minimum_fee_paise": 500},
    {"rail": "upi", "base_rate": "0.000", "p2m_interchange_cap_paise": 0}
  ],
  "clauses": "Amex corporate card transactions incur an additional 50 bps surcharge if issued outside domestic clearing corridors."
}
```
- `query_merchant_contract(payment_method, merchant_id) -> contract_schema`
- `get_tax_rules(date) -> {gst_rate: "0.18", tds_rate: "0.001", effective_from, notes}` —
  **FIX:** `tds_rate` must reflect the 0.1% rate, not 1%, and the tool should carry an
  `effective_from` date so a test can confirm the agent picks the correct rate for a
  transaction dated before vs. after the 2024 rate change, if you choose to model that
  transition; if not modeled, hardcode 0.1% only and say so in a comment.
- `check_settlement_timing(order_timestamp) -> {expected_window, rail_cutoff_rules}` —
  windows expressed in business days, consistent with `src/common/calendar.py`.

No tool writes to the ledger, and no tool marks anything resolved — the agent proposes
an `AgentResolution`, the Layer 3 gatekeeper decides whether to post it.

### 4.3 `src/agent/graph.py` — LangGraph state machine

Nodes: `classify_discrepancy`, `invoke_tool` (max 3 calls), `propose_resolution`
(returns an `AgentResolution`), `gatekeeper_check`, then post or `honest_exception`.

Hard cap: 3 tool calls per record, enforced in the graph, not just prompted.

**`BatchContext` on `DiscrepancyRecord` — required.**
```python
class BatchContext(BaseModel):
    parent_utr: str
    batch_size: int
    sibling_order_ids: list[str]
    aggregate_bank_credit_paise: int

class DiscrepancyRecord(BaseModel):
    # ... existing fields ...
    batch_context: BatchContext | None = None
```
This lets the agent reason about a batch-level flat deduction (e.g. a wire fee spread
across a UTR batch) without giving it cross-record memory.

**Scope note, resolved during Layer 4 code review (2026-08-31): `BatchContext` /
`BATCH_LEVEL_FEE` struck from scope, not built out further.** The schema field and the
`BATCH_LEVEL_FEE` root-cause code exist (`src/agent/discrepancy.py`,
`src/agent/resolution.py`), but every `utr_batch` record in `ground_truth.json` resolves
via `expected_resolution: fast_path` (largest-remainder allocation, Layer 3) — no
generated category ever produces a batch-level fee anomaly that needs agent diagnosis, so
`batch_context` is never populated by either discrepancy-queue builder and
`BATCH_LEVEL_FEE` is unreachable. This is a disclosed, accepted gap: designed for per the
spec above, never exercised because no ground-truth category needs it, and correctly not
claimed as complete anywhere. Building a real batch-level-fee generator category this late
was judged not worth the added generator/ground-truth surface for a code path with no
existing test coverage need; if this changes, wire `batch_context` in
`build_settlement_discrepancy_queue`/`build_unmatched_bank_line_queue` and add a
corresponding `ground_truth.json` category before claiming it's supported.

**Stateless per-record execution — required.** Build a fresh, isolated state dict per
record:
```python
def diagnose_discrepancy(record: DiscrepancyRecord, tools: list) -> AgentResolution:
    initial_state = {
        "record": record, "hop_count": 0, "max_hops": 3,
        "tool_call_history": [], "resolution": None, "status": "PROCESSING",
    }
    return diagnostic_workflow.invoke(initial_state)["resolution"]
```
Add a test processing two records back-to-back where record 1's correct resolution
depends on a fact that would be wrong applied to record 2, confirming no leakage.

**Malformed output handling — required.** Wrap every model response parse in a
try/except targeting the Pydantic validation error specifically. On first failure, feed
the exact error back to the model for one auto-correction attempt. If the second attempt
also fails, route to `honest_exception`, log the raw failure, and continue the batch.

**Critical test before anything else in this layer:** run the agent against the 5
`adversarial_trap` records first. If it force-matches any of them, the guardrail is
broken — fix classify/tool logic before running the rest of the queue.

**Layer 4 acceptance criteria:**
- [ ] All 5 adversarial trap records routed to `honest_exception`, never matched
- [ ] `fee_drift`, `missing_tax_line`, `cutoff_drift` categories resolved via the
      correct tool **and** with the correct `root_cause_code` and `quantified_delta_paise`
      against `ground_truth.json` — not merely "not honest_exception"
- [ ] `orphan` and `short_settlement` records both correctly routed to `honest_exception`
- [ ] `duplicate_credit` records, having reached the agent via the cardinality
      guardrail, are also routed to `honest_exception`, not resolved to either candidate
- [ ] No record consumes more than 3 tool calls
- [ ] A deliberately malformed model payload is caught, retried once, and falls back
      to `honest_exception` without crashing the batch
- [ ] A stateless-execution regression test (record 1's fact does not leak to record 2)
- [ ] `pytest tests/test_agent.py` passes

---

## Layer 5 — Cash forecaster

`src/forecast/cashflow.py` — a rolling projection from posted ledger entries. Full scope,
no time-boxing:

- **Rail-specific settlement windows in business days**: UPI T+1, cards T+2,
  international/other T+3, using `src/common/calendar.py`. Project in-flight funds into
  expected cash-available dates per transaction's payment method.
- Since time is not constrained, it is reasonable to also model: a simple per-rail
  confidence band (e.g. ± a fixed small percentage to represent settlement-day slip),
  and a chart distinguishing "confirmed cash" (already posted to `CASH`) from
  "projected cash" (still in `CASH_IN_TRANSIT_UTR` or `AR_GATEWAY_CLEARING`). Do not go
  further than this — no statistical hazard-rate model, no chargeback probability
  curves, no scenario simulation. This forecaster exists to prove the reconciled ledger
  data is usable downstream, not to be a forecasting product in its own right; adding a
  third dimension of sophistication here would be scope creep relative to what the
  pitch actually needs to demonstrate.

**Acceptance:** produces a 7-day forward projection from the ledger state, using
per-rail, business-day-aware settlement windows, sanity-checked against the known batch
total, with confirmed vs. projected cash visibly distinguished.

---

## Layer 6 — API layer

`src/api/main.py` — FastAPI wrapping: trigger batch run (with a `batch_run_id`), get
reconciliation status, get exception list, get forecast, get trial balance. Full scope,
restored (not cut). Keep it thin — no business logic here that isn't already in the
layers above; this is a thin transport layer over `src/ledger`, `src/matching`, and
`src/forecast`, nothing more. Streamlit (Layer 7) calls this API rather than importing
the modules directly, since both are now in scope.

---

## Layer 7 — Streamlit dashboard

`src/dashboard/app.py`. Views: batch run trigger (supports selecting an existing
`batch_run_id` or entering a live judge-provided seed), match rate summary, honest
exception list (category and notes visible), **full trial balance table** (not just a
pass/fail badge), forecast chart with confirmed-vs-projected cash, and a run selector
that lets both the frozen `challenge_batch_100` run and a live seed run be viewed
side by side, since `batch_run_id` scoping (Layer 3) makes this possible without
destroying either.

---

## Layer 8 — Evaluation harness

`evaluate.py` at project root.

**Deterministic block** — run once, exact numbers, byte-reproducible for a given seed:
```
Fast path resolved:       XX / 100  (target ~60-70%)
Honest exceptions:        XX / 100
False auto-resolutions:   X   (must be 0)
Adversarial traps caught: X / 5   (must be 5/5)
Duplicate credits caught: X / 2   (must be 2/2, not force-matched)
Ledger balance check:     PASS / FAIL
Trial balance:            [full account-by-account table, sums to 0]
Paise round-trip drift:   0 (asserted)
```

**Agent block — FIX (audit item 2), run 3 times, report the range, never a single
number:**
```
Agent resolved (run 1/2/3):        11 / 12 / 13   out of 15
Correct root_cause_code (avg):     91.1% (41/45 across 3 runs)
Correct quantified_delta (avg):    ± tolerance, e.g. within 1% of expected_delta_paise
Honest exceptions from agent path: consistent across all 3 runs? Y/N
```
Every agent invocation logs to `data/agent_runs/<seed>_<run_index>.jsonl`.
`evaluate.py --replay <path>` re-scores from a cached run without calling the model —
this is the offline demo fallback.

This is the only source of truth for any number in the README or the pitch. Never
hand-write these numbers.

**Methodology note: the 3 agent runs were gathered across a multi-day window, not one
sitting.** Groq's free tier caps at 200,000 tokens/day; a full evaluation run touches
all 37 records with `expected_resolution` in `{agent_resolved, honest_exception}`
(measured directly against `ground_truth.json`, not this doc's earlier illustrative
"15" example), and 3 independent live runs of that size exceed one day's budget on a
single free key (observed cost during Layer 4 iteration: ~5,900 tokens per
record-run). Caching the other 2 runs to fit one day was rejected as dishonest — the
whole point of reporting min/median/max is real run-to-run variance (CLAUDE.md
Sec.5), so all 3 must be genuinely independent live runs. `data/agent_runs/<seed>_1.jsonl`,
`_2.jsonl`, `_3.jsonl` will therefore carry timestamps days apart; that gap is this
rate limit, not an inconsistency.

---

## Layer 9 — Packaging, tests, docs

- `docker-compose.yml` — Postgres only. App and Streamlit run directly, not
  containerized, to reduce demo-day failure surface.
- `pytest tests/ -v` — full suite from all layers must pass.
- `README.md` must include: architecture diagram, the "modular monolith, not
  distributed" framing with justification, the LangGraph-vs-Agent-SDK tradeoff note,
  domain equations (MDR/GST 18%/TDS 0.1%), the explicit 194-O modeling-assumption
  sentence (see `CLAUDE.md` §4), the UPI-nil-MDR note, the largest-remainder allocation
  note with the `ROUNDING_DIFFERENCE` account explained, the deterministic scorecard
  **and** the 3-run agent variance block from Layer 8, and the full trial balance table.

---

## Layer 10 — Demo prep

- **Forecast chart demo cutoff:** trigger the frozen batch with
  `as_of=2025-01-20` (dashboard: check "Limit settlement posting to a cutoff
  date", set the date to 2025-01-20) to get a real, non-fabricated
  confirmed/projected split — see the as_of-gated Stage 2 posting fix
  (`src/orchestration/batch_runner.py`). **Verified by an actual triggered
  run against the frozen dataset** (not estimated): with `as_of=2025-01-20`,
  `horizon_days=7` (the dashboard's fixed horizon), the forecast has 39
  confirmed rows (Rs 105,435.50) and 46 projected rows (Rs 132,069.16 total,
  ±5% band), of which 58 rows fall within the 7-day horizon the chart
  actually renders: 39 confirmed (Rs 105,435.50, all bucketed on
  2025-01-20) and 19 projected (Rs 54,467.98) spread across 2025-01-21,
  01-22, 01-23, 01-24, and 01-27. Both bars will be visibly non-empty.
  The frozen dataset's settlement dates span 2025-01-07 to 2025-02-04 —
  a cutoff before 2025-01-07 or after 2025-02-04 produces a degenerate
  all-one-bucket chart (all-projected or all-confirmed respectively) and
  must not be used for the live demo. That same `as_of=2025-01-20` trigger
  reports `fast_path=29, agent_resolved=10, honest_exception=6` (total
  orders 87) — these are **the mid-settlement snapshot numbers for the
  forecast-chart demo only**, lower than the headline **63/20/17 full-batch
  reconciliation numbers** (the unconditional, `as_of=None` run reported
  elsewhere, e.g. in the README scorecard and Layer 3/6 acceptance
  criteria) because Stage 2 is deliberately still gated for orders settling
  after the cutoff. Never quote the two sets of numbers interchangeably —
  reconcile with an unconditional trigger for the headline score, and only
  use the as_of=2025-01-20 trigger for the forecast-chart visual.
- Rehearse the live demo twice end-to-end on the actual presenting machine.
- Record a full backup video in case live Docker/Postgres fails on stage; also confirm
  `evaluate.py --replay` works fully offline as a second-line fallback.
- Write the RCA from something that actually broke during Layer 4 testing (it likely
  will) — do not invent one.
- Final read-through: confirm nothing in the README, pitch script, or code comments
  claims "distributed," "microservices," or cites unverifiable specifics about
  Razorpay's internal stack beyond the one verified Agent Studio / Claude Agent SDK fact.

---

## Submission checklist

- [ ] `docker-compose up` boots Postgres; app and dashboard run via documented commands
- [ ] `pytest tests/ -v` passes, all layers
- [ ] `python evaluate.py` reproduces the deterministic block in the README exactly, and
      the agent block falls within the reported 3-run range
- [ ] `python evaluate.py --replay <cached_run>` works fully offline
- [ ] Honest Exception List correctly isolates all orphans, all short_settlements, all
      duplicate_credits, and all 5 adversarial traps
- [ ] README: architecture, tradeoffs, domain equations, real eval output (both blocks),
      full trial balance
- [ ] 5-minute pitch, live demo plus recorded backup
- [ ] RCA is a real incident with a real fix

---

## On landing the internship, beyond the code

1. **Know your own tradeoffs cold** — modular monolith vs. distributed, LangGraph vs.
   Agent SDK, integer-paise vs. Decimal, largest-remainder allocation, why TDS is 0.1%
   not 1%, why UPI MDR is nil. Two sentences each, no notes.
2. **Lead with the number, not the adjective**, and be precise about which numbers are
   exact and which are a 3-run range — that precision is itself a signal.
3. **The RCA is your best interview story, if it's real.**
