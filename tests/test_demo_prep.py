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
