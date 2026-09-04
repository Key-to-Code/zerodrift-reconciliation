"""Layer 10 tests (docs/plan.md Layer 10: demo prep).

Layer 10 has no formal `- [ ]` acceptance list of its own in the plan --
like Layer 9 (see tests/test_packaging.py's own docstring), it is a bundle
of prose demo-readiness requirements. Restated and approved (2026-09-04)
before writing anything here:

1. The forecast-chart demo cutoff (as_of=2025-01-20, horizon_days=7 on the
   frozen batch) reproduces the exact confirmed/projected numbers
   docs/plan.md Layer 10 documents, via a REAL triggered run against the
   frozen dataset right now -- not copied from the plan's prose (CLAUDE.md
   Sec.1: never write a number unless it was actually produced by running
   the actual code).
2. The plan's warning that a cutoff before 2025-01-07 or after 2025-02-04
   produces a degenerate (all-projected / all-confirmed) chart is a real,
   provable regression, not just prose.
3. `evaluate.py --replay` genuinely works with no model client ever
   constructed -- the offline demo fallback if the venue network/API is
   down.
4. A real RCA exists in README.md, written from something that actually
   broke during Layer 4 testing (the 2026-09-03 Groq daily-token-cap
   incident), cross-checked against real git history so it can't silently
   drift into a fabricated or exaggerated account.
5. Final read-through: nothing in README.md, design.md, or src/**/*.py
   claims "distributed," "microservices," or an unverifiable Razorpay-
   internal fact beyond the one verified Agent Studio/Claude Agent SDK
   claim.

Two Layer 10 bullets are NOT covered here and can't be: rehearsing the live
demo twice on the actual presenting machine, and recording a full backup
video. Both are physical actions on real hardware -- flagged, not faked.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import uuid
from datetime import date
from pathlib import Path

import pytest

from evaluate import format_agent_block_report, resolutions_from_log, score_agent_runs
from src.common.money import to_paise
from src.data.models import BankStatementLine, GatewaySettlement, GroundTruthEntry, InternalOrder
from src.forecast.cashflow import project_cashflow
from src.orchestration.batch_runner import run_batch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_DIR = PROJECT_ROOT / "data" / "challenge_batch_100"
AGENT_RUNS_DIR = PROJECT_ROOT / "data" / "agent_runs"
README = PROJECT_ROOT / "README.md"
DESIGN_MD = PROJECT_ROOT / "design.md"


# ---------------------------------------------------------------------------
# Fixtures: frozen dataset (same pattern as test_forecast.py / test_ledger.py)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def frozen_orders() -> list[InternalOrder]:
    data = json.loads((FROZEN_DIR / "internal_orders.json").read_text(encoding="utf-8"))
    return [InternalOrder.model_validate(d) for d in data]


@pytest.fixture(scope="module")
def frozen_settlements() -> list[GatewaySettlement]:
    data = json.loads((FROZEN_DIR / "gateway_settlement.json").read_text(encoding="utf-8"))
    return [GatewaySettlement.model_validate(d) for d in data]


@pytest.fixture(scope="module")
def frozen_bank_lines() -> list[BankStatementLine]:
    data = json.loads((FROZEN_DIR / "bank_statement.json").read_text(encoding="utf-8"))
    return [BankStatementLine.model_validate(d) for d in data]


# ---------------------------------------------------------------------------
# Test 1: forecast-chart demo cutoff reproduces the plan's real numbers
# (criterion 1)
# ---------------------------------------------------------------------------

def test_forecast_chart_cutoff_reproduces_documented_numbers(
    db_session, batch_run_id, frozen_orders, frozen_settlements, frozen_bank_lines
):
    as_of = date(2025, 1, 20)
    summary = run_batch(db_session, batch_run_id, frozen_orders, frozen_settlements, frozen_bank_lines, as_of=as_of)

    # The mid-settlement snapshot docs/plan.md Layer 10 quotes -- lower than
    # the full-batch 63/20/17 headline because Stage 2 is deliberately
    # gated for anything settling after this cutoff.
    assert summary.total_orders == 87
    assert summary.fast_path_count == 29
    assert summary.agent_resolved_count == 10
    assert summary.honest_exception_count == 6

    result = project_cashflow(db_session, batch_run_id, frozen_orders, frozen_settlements, as_of=as_of, horizon_days=7)

    confirmed = result.filter(result["account_status"] == "confirmed")
    projected = result.filter(result["account_status"] == "projected")
    assert confirmed.height == 39
    assert confirmed["amount_paise"].sum() == to_paise_str("105435.50")
    assert projected.height == 46
    assert projected["amount_paise"].sum() == to_paise_str("132069.16")

    within = result.filter(result["within_horizon"])
    within_confirmed = within.filter(within["account_status"] == "confirmed")
    within_projected = within.filter(within["account_status"] == "projected")
    assert within.height == 58
    assert within_confirmed.height == 39
    assert within_confirmed["amount_paise"].sum() == to_paise_str("105435.50")
    assert within_projected.height == 19
    assert within_projected["amount_paise"].sum() == to_paise_str("54467.98")
    assert sorted(within_projected["expected_cash_date"].unique().to_list()) == [
        date(2025, 1, 21), date(2025, 1, 22), date(2025, 1, 23), date(2025, 1, 24), date(2025, 1, 27),
    ]


def to_paise_str(amount: str) -> int:
    from decimal import Decimal

    return to_paise(Decimal(amount))


# ---------------------------------------------------------------------------
# Test 2: the plan's "don't use these cutoffs" boundary is a real, provable
# degenerate chart, not just a caution in prose (criterion 2)
# ---------------------------------------------------------------------------

def test_forecast_chart_cutoff_before_dataset_range_is_all_projected(
    db_session, batch_run_id, frozen_orders, frozen_settlements, frozen_bank_lines
):
    as_of = date(2025, 1, 6)  # one day before the dataset's earliest settlement_date
    run_batch(db_session, batch_run_id, frozen_orders, frozen_settlements, frozen_bank_lines, as_of=as_of)
    result = project_cashflow(db_session, batch_run_id, frozen_orders, frozen_settlements, as_of=as_of, horizon_days=7)

    confirmed = result.filter(result["account_status"] == "confirmed")
    projected = result.filter(result["account_status"] == "projected")
    assert confirmed.height == 0, "a cutoff before the dataset's own range must produce a degenerate all-projected chart"
    assert projected.height == 87


def test_forecast_chart_cutoff_after_dataset_range_is_all_confirmed(
    db_session, batch_run_id, frozen_orders, frozen_settlements, frozen_bank_lines
):
    as_of = date(2025, 2, 5)  # one day after the dataset's latest settlement_date
    run_batch(db_session, batch_run_id, frozen_orders, frozen_settlements, frozen_bank_lines, as_of=as_of)
    result = project_cashflow(db_session, batch_run_id, frozen_orders, frozen_settlements, as_of=as_of, horizon_days=7)

    confirmed = result.filter(result["account_status"] == "confirmed")
    projected = result.filter(result["account_status"] == "projected")
    assert projected.height == 0, "a cutoff after the dataset's own range must produce a degenerate all-confirmed chart"
    assert confirmed.height == 83


# ---------------------------------------------------------------------------
# Test 3: `evaluate.py --replay` genuinely works with zero network/model
# access -- the documented offline demo fallback (criterion 3)
# ---------------------------------------------------------------------------

def test_evaluate_replay_works_fully_offline():
    log_paths = [AGENT_RUNS_DIR / f"frozen_{i}.jsonl" for i in (1, 2, 3)]
    for p in log_paths:
        assert p.exists(), f"{p} missing -- the agent-block sweep this test replays must already exist"

    _, ground_truth_data = _load_ground_truth()
    run_resolutions = [resolutions_from_log(p) for p in log_paths]
    expected_output = format_agent_block_report(score_agent_runs(run_resolutions, ground_truth_data))

    env = {
        k: v
        for k, v in __import__("os").environ.items()
        if k.upper() not in ("GROQ_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")
    }
    proc = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "evaluate.py"), "--replay", str(AGENT_RUNS_DIR / "frozen"), "--runs", "3"],
        cwd=PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, (
        f"evaluate.py --replay must succeed with no model API key present at all -- if it needed a live "
        f"model client this would fail fast instead. stderr:\n{proc.stderr}"
    )
    assert proc.stdout.strip() == expected_output.strip()


def _load_ground_truth() -> tuple[None, list[GroundTruthEntry]]:
    data = json.loads((FROZEN_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    return None, [GroundTruthEntry.model_validate(d) for d in data]


# ---------------------------------------------------------------------------
# Test 4: README's RCA section describes a real incident, cross-checked
# against real git history (criterion 4)
# ---------------------------------------------------------------------------

def _readme_text() -> str:
    return README.read_text(encoding="utf-8")


def _git_commit_subject(commit_ish: str) -> str:
    result = subprocess.run(
        ["git", "log", "-1", "--format=%s", commit_ish], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def _git_commit_touches_file(commit_ish: str, path: str) -> bool:
    result = subprocess.run(
        ["git", "show", "--stat", "--format=", commit_ish], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    )
    return path in result.stdout


def test_readme_rca_section_exists():
    text = _readme_text()
    assert re.search(r"^##.*RCA", text, re.MULTILINE), "README.md must have an RCA section"


def test_readme_rca_cross_checked_against_real_git_history():
    """The two commits that actually fixed the real Layer 4 incident."""
    fix_commit = "fc1d159"
    preflight_commit = "e7ace91"

    assert _git_commit_touches_file(fix_commit, "src/agent/rate_limiter.py")
    assert _git_commit_touches_file(preflight_commit, "src/orchestration/batch_runner.py")
    fix_subject = _git_commit_subject(fix_commit)
    preflight_subject = _git_commit_subject(preflight_commit)
    assert "daily-token-budget" in fix_subject.lower() or "daily token" in fix_subject.lower()
    assert "pre-flight" in preflight_subject.lower()

    text = _readme_text()
    assert fix_commit in text, "README RCA must cite the real fix commit hash, not describe it vaguely"
    assert preflight_commit in text, "README RCA must cite the real pre-flight-check commit hash"
    assert "Groq" in text
    assert "200,000" in text or "200000" in text
    assert "429" in text, "the RCA should name the real error (a Groq 429) rather than a generic description"


# ---------------------------------------------------------------------------
# Test 4b-4e: the four additional real incidents added to the RCA section
# (2026-09-04, at the user's request), each cross-checked against real git
# history or, for the one with no commit, reproduced fresh right here.
# ---------------------------------------------------------------------------

def test_readme_layer4_marathon_incident_cross_checked_against_git_history():
    commit = "f6d0f49"
    assert _git_commit_touches_file(commit, "src/agent/graph.py")
    assert _git_commit_touches_file(commit, "scripts/diagnose_one.py")

    text = _readme_text()
    assert commit in text
    assert "ORD1069" in text, "the concrete AMEX/INTL priority example must be named, not described vaguely"


def test_readme_gatekeeper_incident_cross_checked_against_git_history():
    commit = "00c745c"
    assert _git_commit_touches_file(commit, "src/agent/graph.py")
    subject = _git_commit_subject(commit)
    assert "gatekeeper" in subject.lower()

    text = _readme_text()
    assert commit in text
    assert "37" in text and "0 of 37" in text, "the offline-replay-found-zero-flips claim must state the real count"


def test_readme_refund_clawback_incident_cross_checked_against_git_history(frozen_orders, frozen_settlements):
    commit = "ab6feed"
    assert _git_commit_touches_file(commit, "src/ledger/journal.py")

    settlements_by_id = {s.order_id: s for s in frozen_settlements}
    refund_orders = [o for o in frozen_orders if o.refund_amount is not None]
    assert len(refund_orders) == 3
    gross_total = sum((settlements_by_id[o.order_id].gross_amount for o in refund_orders), start=__import__("decimal").Decimal("0"))
    refund_total = sum((o.refund_amount for o in refund_orders), start=__import__("decimal").Decimal("0"))
    mdr_total = sum((settlements_by_id[o.order_id].mdr for o in refund_orders), start=__import__("decimal").Decimal("0"))

    text = _readme_text()
    assert commit in text
    assert f"{gross_total:,.2f}" in text or str(gross_total) in text, (
        f"README must state the real combined gross ({gross_total}), freshly recomputed from the frozen dataset"
    )
    assert f"{refund_total:,.2f}" in text or str(refund_total) in text
    assert f"{mdr_total:,.2f}" in text or str(mdr_total) in text


def test_readme_stress_test_incident_numbers_match_a_fresh_run():
    """No commit exists for this one -- it never needed a code change. So
    instead of cross-checking against git, this test reproduces the whole
    thing live, right now, and asserts README's numbers match a fresh run
    -- exactly the standard CLAUDE.md Sec.1 sets for every other number in
    this project."""
    import math
    import uuid

    from sqlalchemy import create_engine, func, select

    from src.agent.graph import AGENT_LOGIC_VERSION
    from src.agent.discrepancy import build_settlement_discrepancy_queue, build_unmatched_bank_line_queue
    from src.agent.rate_limiter import AgentRateLimitedError, DAILY_TOKEN_BUDGET, daily_token_tracker
    from src.agent.run_log import average_real_tokens_per_live_call, count_live_calls_needed
    from src.data.generator import generate_batch
    from src.ledger.models import JournalEntry, ReconciliationMatch, get_sessionmaker, reset_schema
    from src.matching.schema import bank_lines_to_frame, orders_to_frame, settlements_to_frame
    from src.orchestration.batch_runner import DEFAULT_AGENT_CACHE_PATH, run_batch

    batch = generate_batch(num_records=1000, seed=999)
    assert len(batch.orders) == 780

    orders_df = orders_to_frame(batch.orders)
    settlements_df = settlements_to_frame(batch.settlements)
    bank_df = bank_lines_to_frame(batch.bank_lines)
    settlement_records = build_settlement_discrepancy_queue(orders_df, settlements_df, bank_df)
    unmatched_records = build_unmatched_bank_line_queue(orders_df, settlements_df, bank_df)
    all_records = settlement_records + unmatched_records
    assert len(all_records) == 370

    avg_tokens = average_real_tokens_per_live_call([DEFAULT_AGENT_CACHE_PATH])
    n_live = count_live_calls_needed(all_records, DEFAULT_AGENT_CACHE_PATH, AGENT_LOGIC_VERSION)
    assert n_live == 370
    estimated_tokens = math.ceil(n_live * avg_tokens)
    assert estimated_tokens == 2_546_774

    engine = create_engine(
        "postgresql+psycopg://postgres:postgres@localhost:5433/finance_controller_test", future=True
    )
    reset_schema(engine)
    session = get_sessionmaker(engine)()
    daily_token_tracker.reset_for_testing()
    try:
        with pytest.raises(AgentRateLimitedError):
            run_batch(session, uuid.uuid4(), batch.orders, batch.settlements, batch.bank_lines)
        n_entries = session.execute(select(func.count()).select_from(JournalEntry)).scalar_one()
        n_matches = session.execute(select(func.count()).select_from(ReconciliationMatch)).scalar_one()
        assert n_entries == 0
        assert n_matches == 0
    finally:
        session.close()
        daily_token_tracker.reset_for_testing()

    text = _readme_text()
    assert "780" in text and "370" in text
    assert "2,546,774" in text or "2546774" in text
    assert "6,883" in text or "6883" in text
    assert estimated_tokens / DAILY_TOKEN_BUDGET > 12 and estimated_tokens / DAILY_TOKEN_BUDGET < 13
    assert "13" in text, "the real ceil(days-to-clear) figure must be stated"


def test_readme_batch_run_persistence_incident_cross_checked_against_git_history():
    commit = "663ddf2"
    assert _git_commit_touches_file(commit, "src/ledger/models.py")
    assert _git_commit_touches_file(commit, "src/api/main.py")
    subject = _git_commit_subject(commit)
    assert "persist" in subject.lower() and "batch_run" in subject.lower()

    text = _readme_text()
    assert commit in text
    assert "batch_run_recipes" in text
    assert "404" in text, "the real symptom (a 404 on an existing run) must be named"


def test_batch_run_recipe_table_actually_backs_the_status_endpoint(pg_engine):
    """Independent re-proof of the incident 6 claim, not just a citation:
    delete the persisted recipe row for real ledger data and confirm the
    status lookup genuinely depends on it (no leftover in-memory
    fallback), then confirm re-inserting the row alone recovers it. Same
    dependency-override pattern as tests/test_api.py's own `client`
    fixture -- mirrors
    test_status_endpoint_depends_only_on_the_persisted_recipe_row_not_any_process_state,
    re-run here rather than merely cited."""
    import uuid as uuid_module

    from fastapi.testclient import TestClient

    from src.api.main import app, get_session
    from src.ledger.models import BatchRunRecipe, get_sessionmaker

    session_factory = get_sessionmaker(pg_engine)

    def _override_get_session():
        session = session_factory()
        try:
            yield session
        finally:
            session.rollback()
            session.close()

    app.dependency_overrides[get_session] = _override_get_session
    try:
        with TestClient(app) as client:
            resp = client.post("/batch-runs", json={"source": "frozen"})
            assert resp.status_code == 200
            batch_run_id = resp.json()["batch_run_id"]
            assert client.get(f"/batch-runs/{batch_run_id}/status").status_code == 200

            session = session_factory()
            row = session.get(BatchRunRecipe, uuid_module.UUID(batch_run_id))
            session.delete(row)
            session.commit()
            session.close()

            assert client.get(f"/batch-runs/{batch_run_id}/status").status_code == 404, (
                "with the recipe row gone, this must 404 -- proves no hidden in-memory registry is covering for it"
            )

            session = session_factory()
            session.add(
                BatchRunRecipe(batch_run_id=uuid_module.UUID(batch_run_id), source="frozen", seed=None, records=100)
            )
            session.commit()
            session.close()

            resp = client.get(f"/batch-runs/{batch_run_id}/status")
            assert resp.status_code == 200, (
                "re-inserting the recipe row alone must recover access to the still-real ledger data"
            )
            assert resp.json()["total"] > 0
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# Test 5: final read-through -- no forbidden architecture language anywhere
# in README.md, design.md, or src/**/*.py (criterion 5)
# ---------------------------------------------------------------------------

def _all_scanned_files() -> list[Path]:
    return [README, DESIGN_MD] + sorted((PROJECT_ROOT / "src").rglob("*.py"))


def test_no_bare_microservice_claim_anywhere():
    for path in _all_scanned_files():
        text = path.read_text(encoding="utf-8")
        assert "microservice" not in text.lower(), f"{path} must never describe this system as microservices"


def test_no_unhedged_distributed_claim_anywhere():
    """'distributed' is allowed only inside an explicit negation ('not
    distributed') or an explicit hypothetical contrast ('a distributed
    version of this system would ...') -- never as a bare claim that the
    system itself is distributed."""
    for path in _all_scanned_files():
        text = path.read_text(encoding="utf-8")
        for m in re.finditer(r"distributed", text, re.IGNORECASE):
            i = m.start()
            preceding = text[max(0, i - 15) : i].lower()
            following = text[m.end() : m.end() + 10].lower()
            is_negated = "not " in preceding
            is_hypothetical = following.startswith(" version")
            assert is_negated or is_hypothetical, (
                f"{path}: unhedged 'distributed' claim at {text[max(0, i - 40) : i + 40]!r} -- CLAUDE.md Sec.2"
            )


def test_kafka_spark_broker_only_mentioned_as_explicitly_absent():
    """CLAUDE.md Sec.7 forbids ADDING these, not the word itself -- the
    README is allowed (and expected, Layer 9) to disclose that they were
    deliberately not used. Every occurrence must sit in a clear negation
    ('no Kafka', 'not ... message broker', etc.)."""
    for path in _all_scanned_files():
        text = path.read_text(encoding="utf-8")
        for term in ("kafka", "spark", "message broker"):
            for m in re.finditer(re.escape(term), text, re.IGNORECASE):
                i = m.start()
                preceding = text[max(0, i - 30) : i].lower()
                assert "no " in preceding or "not " in preceding, (
                    f"{path}: {term!r} mentioned without a clear negation nearby at "
                    f"{text[max(0, i - 40) : i + 40]!r} -- CLAUDE.md Sec.7"
                )


def test_only_verified_razorpay_claim_is_agent_studio_claude_sdk():
    """The one Razorpay-internal claim this project is allowed to make,
    anywhere, is that Agent Studio runs on Anthropic's Claude Agent SDK
    (CLAUDE.md Sec.2). This is a light guard, not exhaustive: it only
    catches 'Razorpay['s]-internal ...' or 'Razorpay['s] ... stack'
    immediately adjacent -- a tight window, not a wide one, since a wide
    window over the README's own Mermaid diagram picks up unrelated
    matches like 'internal_orders.json'."""
    text = _readme_text()
    for m in re.finditer(r"Razorpay('s)?[\s-]{1,3}(internal|stack)", text, re.IGNORECASE):
        window = text[max(0, m.start() - 20) : m.end() + 220]
        assert "Agent Studio" in window and ("Claude Agent SDK" in window or "Anthropic" in window), (
            f"unverifiable Razorpay-internal claim near: {window!r} -- CLAUDE.md Sec.2"
        )
