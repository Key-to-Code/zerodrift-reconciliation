"""Layer 9 tests (docs/plan.md Layer 9: packaging, tests, docs).

Layer 9 has no unit-level "acceptance checkbox" list of its own in the plan
-- it is a bundle of packaging/doc requirements, verified here two ways:

1. Structural checks (docker-compose stays Postgres-only, README contains
   the required sections) -- these prove the required *content* exists.
2. Anti-fabrication checks (CLAUDE.md Sec.1: "Never write a number into
   README.md ... unless that exact number was produced by actually running
   the actual code against the actual dataset") -- these re-run the real
   pipeline against the real frozen dataset (same conftest.py Postgres
   fixtures test_ledger.py/test_evaluate.py already use, no mocking) and
   assert the numbers embedded in README.md match exactly. If the README
   drifts from a re-run, these fail loudly instead of a diagram silently
   going stale.

The agent-block scorecard is the one section that cannot be produced
today: it requires 3 independent live agent sweeps, gathered across
multiple days under Groq's free-tier daily token cap (evaluate.py's own
module docstring, docs/plan.md Layer 8's methodology note). Test 5 below
skips -- not fails -- until data/agent_runs/frozen_1/2/3.jsonl all exist,
then activates for real.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from evaluate import (
    build_discrepancy_queue,
    format_agent_block_report,
    format_deterministic_report,
    load_frozen_dataset,
    resolutions_from_log,
    run_deterministic_block,
    score_agent_runs,
)
from src.data.generator import GST_RATE, TDS_RATE

PROJECT_ROOT = Path(__file__).resolve().parents[1]
README = PROJECT_ROOT / "README.md"
DOCKER_COMPOSE = PROJECT_ROOT / "docker-compose.yml"
AGENT_RUNS_DIR = PROJECT_ROOT / "data" / "agent_runs"


def _readme_text() -> str:
    return README.read_text(encoding="utf-8")


def _extract_marked_block(text: str, marker: str) -> str:
    """Pulls the fenced code block between
    <!-- marker:START --> ... ``` ... ``` ... <!-- marker:END --> in
    README.md, stripping the fence itself. Raises with a clear message if
    the marker pair (and a fenced block inside it) isn't present -- a
    missing scorecard section should fail loudly, not be treated as an
    empty string that happens to not-match anything."""
    pattern = rf"<!-- {re.escape(marker)}:START -->\s*```(?:text)?\n(.*?)\n```\s*<!-- {re.escape(marker)}:END -->"
    match = re.search(pattern, text, re.DOTALL)
    if match is None:
        raise AssertionError(f"README.md has no {marker} block between its START/END markers")
    return match.group(1)


# ---------------------------------------------------------------------------
# Test 1: docker-compose.yml stays Postgres-only (criterion 1)
# ---------------------------------------------------------------------------

def test_docker_compose_defines_postgres_service_only():
    text = DOCKER_COMPOSE.read_text(encoding="utf-8")
    lines = text.splitlines()

    services_idx = next(i for i, line in enumerate(lines) if line.strip() == "services:")
    volumes_idx = next(
        (i for i, line in enumerate(lines) if line.strip() == "volumes:" and i > services_idx), len(lines)
    )
    service_block = lines[services_idx + 1 : volumes_idx]

    # top-level service names are 2-space indented, followed by ':' -- a
    # deeper-indented line (image:, ports:, etc.) is not a new service.
    service_names = [
        line.strip().rstrip(":")
        for line in service_block
        if line.startswith("  ") and not line.startswith("   ") and line.strip().endswith(":")
    ]
    assert service_names == ["postgres"], f"expected only a 'postgres' service, found {service_names}"

    postgres_block_text = "\n".join(service_block)
    assert "image: postgres:" in postgres_block_text


# ---------------------------------------------------------------------------
# Test 2: README contains the required architecture/domain content, and its
# numeric claims match the actual named constants (criteria 3a-3g)
# ---------------------------------------------------------------------------

def test_readme_contains_architecture_diagram():
    text = _readme_text()
    assert "```mermaid" in text, "README.md must include a Mermaid architecture diagram"


def test_readme_states_modular_monolith_framing():
    text = _readme_text()
    assert "modular monolith" in text
    assert "not distributed" in text or "not a distributed" in text


def test_readme_states_langgraph_vs_agent_sdk_tradeoff():
    text = _readme_text()
    assert "LangGraph" in text
    assert "Agent SDK" in text


def test_readme_domain_equations_match_actual_generator_constants():
    """Guards against exactly the drift CLAUDE.md exists to prevent: the
    README's prose rates must match src/data/generator.py's real GST_RATE /
    TDS_RATE constants, not a hand-typed string that could silently go
    stale if the rate ever changed in code."""
    text = _readme_text()
    gst_percent = f"{GST_RATE * 100:.0f}%"
    tds_percent = f"{TDS_RATE * 100:.1f}%"
    assert gst_percent == "18%"
    assert tds_percent == "0.1%"
    assert gst_percent in text, f"README must state the real GST_RATE ({gst_percent}), not a stale figure"
    assert tds_percent in text, f"README must state the real TDS_RATE ({tds_percent}), not a stale figure"
    # The old pre-2024-10-01 rate may only appear as part of "0.1%" itself,
    # or as explicit historical context ("reduced from 1% ...", matching the
    # phrasing already used by src/agent/tools.py's own get_tax_rules notes
    # field) -- never as a standalone claim that TDS currently is 1%.
    stray_positions = [
        i
        for i in (m.start() for m in re.finditer(r"1%", text))
        if text[max(0, i - 2) : i] != "0." and not text[max(0, i - 13) : i].endswith("reduced from ")
    ]
    assert not stray_positions, (
        f"README contains a stray '1%' outside '0.1%'/'reduced from 1%' context at "
        f"{[text[max(0, i - 20) : i + 5] for i in stray_positions]} -- CLAUDE.md Sec.4"
    )


def test_readme_states_194o_modeling_assumption():
    text = _readme_text()
    assert "194-O" in text
    assert "e-commerce participant" in text
    assert "aggregator" in text
    assert "does not resolve" in text or "doesn't resolve" in text


def test_readme_states_upi_nil_mdr():
    text = _readme_text()
    assert "UPI" in text
    assert "nil" in text
    assert "MDR" in text


def test_readme_states_largest_remainder_allocation_and_rounding_account():
    text = _readme_text()
    assert "largest-remainder" in text or "largest remainder" in text
    assert "ROUNDING_DIFFERENCE" in text


# ---------------------------------------------------------------------------
# Tests 3-4: the deterministic scorecard and trial balance embedded in
# README.md match a real run against the frozen dataset, right now
# (criterion 3h deterministic half, 3i) -- CLAUDE.md Sec.1 anti-fabrication.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def live_deterministic_report_text():
    import uuid

    from src.ledger.models import ensure_schema_exists, get_engine, get_sessionmaker

    orders, settlements, bank_lines, ground_truth = load_frozen_dataset()
    engine = get_engine()
    ensure_schema_exists(engine)
    session = get_sessionmaker(engine)()
    try:
        report = run_deterministic_block(session, uuid.uuid4(), orders, settlements, bank_lines, ground_truth)
    finally:
        session.close()
    return format_deterministic_report(report)


def test_readme_deterministic_scorecard_matches_real_evaluate_run(live_deterministic_report_text):
    embedded = _extract_marked_block(_readme_text(), "SCORECARD:DETERMINISTIC")
    assert embedded.strip() == live_deterministic_report_text.strip(), (
        "README.md's deterministic scorecard does not match a fresh run of evaluate.py against the "
        "frozen dataset -- re-run `python evaluate.py --skip-agent-block` and paste the real output "
        "(CLAUDE.md Sec.1: never hand-write a number here)"
    )


def test_readme_trial_balance_rows_present_and_correct(live_deterministic_report_text):
    embedded = _extract_marked_block(_readme_text(), "SCORECARD:DETERMINISTIC")
    for account in (
        "CASH", "CASH_IN_TRANSIT_UTR", "AR_GATEWAY_CLEARING", "REVENUE_GROSS",
        "MDR_EXPENSE", "GST_ITC_RECEIVABLE", "TDS_194O_CREDIT", "TOTAL",
    ):
        assert re.search(rf"^\s*{account}\b", embedded, re.MULTILINE), f"trial balance row for {account} missing"
    total_line = next(line for line in embedded.splitlines() if line.strip().startswith("TOTAL"))
    assert "net=             0" in total_line or "net=            0" in total_line or re.search(
        r"net=\s+0\b", total_line
    ), f"TOTAL row must net to exactly zero, got: {total_line!r}"


# ---------------------------------------------------------------------------
# Test 5: the agent-block scorecard, once the real 3-run sweep exists
# (criterion 3h agent half) -- skipped, not failed, until then.
# ---------------------------------------------------------------------------

def test_readme_agent_block_matches_replayed_sweep():
    log_paths = [AGENT_RUNS_DIR / f"frozen_{i}.jsonl" for i in (1, 2, 3)]
    if not all(p.exists() for p in log_paths):
        pytest.skip(
            "agent-block 3-run sweep not complete yet (data/agent_runs/frozen_1/2/3.jsonl) -- "
            "gathered across multiple days per evaluate.py's Groq daily-token-cap note"
        )
    _, _, _, ground_truth = load_frozen_dataset()
    run_resolutions = [resolutions_from_log(p) for p in log_paths]
    live_report_text = format_agent_block_report(score_agent_runs(run_resolutions, ground_truth))

    embedded = _extract_marked_block(_readme_text(), "SCORECARD:AGENT")
    assert embedded.strip() == live_report_text.strip(), (
        "README.md's agent-block scorecard does not match replaying the real "
        "data/agent_runs/frozen_1/2/3.jsonl logs -- CLAUDE.md Sec.1"
    )


# ---------------------------------------------------------------------------
# Test 6: the two blocks stay visually/structurally separate in README.md
# (CLAUDE.md Sec.5 -- never merge the exact deterministic numbers with the
# agent's min/median/max range into one misleadingly precise figure)
# ---------------------------------------------------------------------------

def test_readme_deterministic_and_agent_scorecards_stay_separate():
    text = _readme_text()
    det_block = _extract_marked_block(text, "SCORECARD:DETERMINISTIC")
    assert "Agent block" not in det_block
    assert "min / median / max" not in det_block
