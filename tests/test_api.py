"""Layer 6 tests for the FastAPI transport layer (src/api/main.py) and its
orchestration layer (src/orchestration/batch_runner.py). Written before
implementation per CLAUDE.md's build protocol.

Two groups of test live here, and it matters which is which:

- HTTP tests (via fastapi.testclient.TestClient) exercise the frozen
  challenge_batch_100 dataset end-to-end through the real API surface. They
  use the real diagnose_or_replay default (no dependency override for
  get_diagnose_fn) -- this is safe and deterministic because every one of
  the 37 non-fast-path frozen records is already cached at the current
  AGENT_LOGIC_VERSION (verified before writing this file: 37/37 cached,
  zero stale), so these tests make zero live model calls and zero network
  requests. Test 14 is the explicit regression guard for that claim.
- Orchestrator-level tests (9-13) call src.orchestration.batch_runner.run_batch
  directly against small synthetic records, with an injected stub
  diagnose_fn -- these prove the new category -> ledger-posting mapping
  itself (approved design, see the Layer 6 planning conversation), not
  anything about live agent behavior.

Runs against a real Postgres test database (tests/conftest.py) -- CLAUDE.md
forbids mocking the ledger layer.
"""
from __future__ import annotations

import json
import uuid
from decimal import Decimal
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from src.agent.discrepancy import DiscrepancyRecord
from src.agent.resolution import AgentResolution
from src.data.models import BankStatementLine, GatewaySettlement, GroundTruthEntry, InternalOrder
from src.ledger.journal import (
    AR_GATEWAY_CLEARING,
    CASH,
    MDR_EXPENSE,
    REVENUE_GROSS,
    SUSPENSE_UNRESOLVED,
    assert_all_entries_have_balanced_lines,
    trial_balance,
)
from src.ledger.models import Account, JournalEntry, JournalLine, ReconciliationMatch, get_sessionmaker
from src.orchestration.batch_runner import run_batch

FROZEN_DIR = Path(__file__).resolve().parents[1] / "data" / "challenge_batch_100"

FROZEN_FAST_PATH_COUNT = 63
FROZEN_AGENT_RESOLVED_COUNT = 20
FROZEN_HONEST_EXCEPTION_COUNT = 17
FROZEN_TOTAL_GROUND_TRUTH_ENTRIES = 100


# ---------------------------------------------------------------------------
# Fixtures: frozen dataset (same pattern as test_ledger.py/test_forecast.py/
# test_agent.py), plus a TestClient wired to the Postgres test database.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def frozen_ground_truth() -> list[GroundTruthEntry]:
    data = json.loads((FROZEN_DIR / "ground_truth.json").read_text(encoding="utf-8"))
    return [GroundTruthEntry.model_validate(d) for d in data]


@pytest.fixture()
def client(pg_engine):
    from src.api.main import app, get_diagnose_fn, get_session

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
        with TestClient(app) as c:
            yield c
    finally:
        app.dependency_overrides.clear()


def _stub_diagnose_fn():
    def _fn(record: DiscrepancyRecord) -> tuple[AgentResolution, dict]:
        return (
            AgentResolution(
                root_cause_code="UNRESOLVED", quantified_delta_paise=0, evidence_tool_calls=[], confidence_note="stub"
            ),
            {"hop_count": 0, "tool_call_history": [], "status": "HONEST_EXCEPTION", "raw_failure": None, "tokens_used": 0},
        )

    return _fn


# ---------------------------------------------------------------------------
# Test 1 -- app exposes the 5 required routes (AC1)
# ---------------------------------------------------------------------------

def test_app_exposes_expected_routes():
    from src.api.main import app

    paths = {route.path for route in app.routes}
    assert "/batch-runs" in paths
    assert "/batch-runs/{batch_run_id}/status" in paths
    assert "/batch-runs/{batch_run_id}/exceptions" in paths
    assert "/batch-runs/{batch_run_id}/trial-balance" in paths
    assert "/batch-runs/{batch_run_id}/forecast" in paths


# ---------------------------------------------------------------------------
# Test 2 -- triggering a frozen batch run posts the full pipeline and writes
# a reconciliation_matches row for every real order plus every unmatched
# bank line (100 total, matching ground_truth.json exactly) (AC2)
# ---------------------------------------------------------------------------

def test_trigger_batch_run_frozen_posts_full_pipeline_and_reconciliation_rows(client, db_session):
    resp = client.post("/batch-runs", json={"source": "frozen"})
    assert resp.status_code == 200
    body = resp.json()
    batch_run_id = uuid.UUID(body["batch_run_id"])

    assert body["total_orders"] == 87
    assert body["total_unmatched_bank_lines"] == 13
    assert body["fast_path_count"] == FROZEN_FAST_PATH_COUNT
    assert body["agent_resolved_count"] == FROZEN_AGENT_RESOLVED_COUNT
    assert body["honest_exception_count"] == FROZEN_HONEST_EXCEPTION_COUNT

    match_count = db_session.execute(
        select(ReconciliationMatch).where(ReconciliationMatch.batch_run_id == batch_run_id)
    ).scalars().all()
    assert len(match_count) == FROZEN_TOTAL_GROUND_TRUTH_ENTRIES

    assert_all_entries_have_balanced_lines(db_session, batch_run_id)
    tb = trial_balance(db_session, batch_run_id)
    total_row = [r for r in tb.to_dicts() if r["account_code"] == "TOTAL"][0]
    assert total_row["net_balance_paise"] == 0


# ---------------------------------------------------------------------------
# Test 3 -- two triggered runs coexist independently (AC3)
# ---------------------------------------------------------------------------

def test_two_triggered_batch_runs_do_not_cross_contaminate(client, db_session):
    resp1 = client.post("/batch-runs", json={"source": "frozen"})
    resp2 = client.post("/batch-runs", json={"source": "frozen"})
    id1 = resp1.json()["batch_run_id"]
    id2 = resp2.json()["batch_run_id"]
    assert id1 != id2

    for batch_run_id in (id1, id2):
        status_resp = client.get(f"/batch-runs/{batch_run_id}/status")
        assert status_resp.json()["total"] == FROZEN_TOTAL_GROUND_TRUTH_ENTRIES
        tb_resp = client.get(f"/batch-runs/{batch_run_id}/trial-balance")
        total_row = [r for r in tb_resp.json() if r["account_code"] == "TOTAL"][0]
        assert total_row["net_balance_paise"] == 0


# ---------------------------------------------------------------------------
# Test 4 -- reconciliation status matches Layer 2/4's already-proven counts
# (AC4), and totals reconcile exactly against ground_truth.json's 100
# entries (63+20+17=100; 17 honest_exception = 4 order-based [short_settlement
# + duplicate_credit] + 13 bank-only [orphan + adversarial_trap])
# ---------------------------------------------------------------------------

def test_reconciliation_status_matches_known_frozen_counts(client):
    batch_run_id = client.post("/batch-runs", json={"source": "frozen"}).json()["batch_run_id"]
    status = client.get(f"/batch-runs/{batch_run_id}/status").json()
    assert status["fast_path"] == FROZEN_FAST_PATH_COUNT
    assert status["agent_resolved"] == FROZEN_AGENT_RESOLVED_COUNT
    assert status["honest_exception"] == FROZEN_HONEST_EXCEPTION_COUNT
    assert status["total"] == FROZEN_TOTAL_GROUND_TRUTH_ENTRIES


# ---------------------------------------------------------------------------
# Test 5 -- exception list returns exactly the 17 honest_exception records,
# with category (discrepancy_reason) and notes visible, nothing else (AC5)
# ---------------------------------------------------------------------------

def test_exception_list_returns_only_the_17_honest_exceptions(client, frozen_ground_truth):
    batch_run_id = client.post("/batch-runs", json={"source": "frozen"}).json()["batch_run_id"]
    exceptions = client.get(f"/batch-runs/{batch_run_id}/exceptions").json()
    assert len(exceptions) == FROZEN_HONEST_EXCEPTION_COUNT

    expected_ids = {e.order_id for e in frozen_ground_truth if e.expected_resolution == "honest_exception"}
    returned_ids = {e["order_id"] for e in exceptions}
    assert returned_ids == expected_ids

    for e in exceptions:
        assert e["status"] == "honest_exception"
        assert e["confidence_note"]  # category/reason + note must be visible, never blank


# ---------------------------------------------------------------------------
# Test 6 -- trial balance served over HTTP matches a direct journal.py call
# (AC6)
# ---------------------------------------------------------------------------

def test_trial_balance_endpoint_matches_direct_journal_call(client, db_session):
    batch_run_id = client.post("/batch-runs", json={"source": "frozen"}).json()["batch_run_id"]
    http_rows = client.get(f"/batch-runs/{batch_run_id}/trial-balance").json()
    direct_rows = trial_balance(db_session, uuid.UUID(batch_run_id)).to_dicts()
    assert http_rows == direct_rows


# ---------------------------------------------------------------------------
# Test 7 -- forecast served over HTTP matches a direct cashflow.py call
# (AC7)
# ---------------------------------------------------------------------------

def test_forecast_endpoint_matches_direct_cashflow_call(client, db_session):
    from datetime import date as _date

    from src.forecast.cashflow import project_cashflow

    batch_run_id = client.post("/batch-runs", json={"source": "frozen"}).json()["batch_run_id"]
    orders = [
        InternalOrder.model_validate(d)
        for d in json.loads((FROZEN_DIR / "internal_orders.json").read_text(encoding="utf-8"))
    ]
    settlements = [
        GatewaySettlement.model_validate(d)
        for d in json.loads((FROZEN_DIR / "gateway_settlement.json").read_text(encoding="utf-8"))
    ]

    http_rows = client.get(
        f"/batch-runs/{batch_run_id}/forecast", params={"as_of": "2025-01-06", "horizon_days": 7}
    ).json()
    direct = project_cashflow(
        db_session, uuid.UUID(batch_run_id), orders, settlements, as_of=_date(2025, 1, 6), horizon_days=7
    )

    assert len(http_rows) == len(direct)
    direct_by_order = {row["order_id"]: row for row in direct.to_dicts()}
    for http_row in http_rows:
        d = direct_by_order[http_row["order_id"]]
        assert http_row["amount_paise"] == d["amount_paise"]
        assert http_row["account_status"] == d["account_status"]
        assert http_row["low_paise"] == d["low_paise"]
        assert http_row["high_paise"] == d["high_paise"]


# ---------------------------------------------------------------------------
# Test 8 -- unknown batch_run_id returns a clean 404 on every GET endpoint,
# never a 500 or a silently empty result (AC9)
# ---------------------------------------------------------------------------

def test_unknown_batch_run_id_returns_404_on_every_get_endpoint(client):
    unknown_id = str(uuid.uuid4())
    assert client.get(f"/batch-runs/{unknown_id}/status").status_code == 404
    assert client.get(f"/batch-runs/{unknown_id}/exceptions").status_code == 404
    assert client.get(f"/batch-runs/{unknown_id}/trial-balance").status_code == 404
    assert client.get(f"/batch-runs/{unknown_id}/forecast", params={"as_of": "2025-01-06"}).status_code == 404


# ---------------------------------------------------------------------------
# Test 9 -- all 5 adversarial traps and both duplicate_credits, reached
# end-to-end through the API, are still routed to honest_exception -- the
# new orchestration layer must not weaken Layer 2/4's guardrails (AC10)
# ---------------------------------------------------------------------------

def test_adversarial_traps_and_duplicate_credits_routed_honest_exception_via_api(client, frozen_ground_truth):
    batch_run_id = client.post("/batch-runs", json={"source": "frozen"}).json()["batch_run_id"]
    exceptions_by_id = {e["order_id"]: e for e in client.get(f"/batch-runs/{batch_run_id}/exceptions").json()}

    traps = [e for e in frozen_ground_truth if e.category == "adversarial_trap"]
    duplicates = [e for e in frozen_ground_truth if e.category == "duplicate_credit"]
    assert len(traps) == 5
    assert len(duplicates) == 2

    for gt in traps + duplicates:
        assert gt.order_id in exceptions_by_id, f"{gt.order_id} ({gt.category}) missing from exception list"
        assert exceptions_by_id[gt.order_id]["status"] == "honest_exception"


# ---------------------------------------------------------------------------
# Test 10 -- orchestrator: agent_resolved with a real root cause posts like
# clean_match, using the settlement's REAL figures (never a "corrected" one)
# ---------------------------------------------------------------------------

def test_orchestrator_agent_resolved_posts_like_clean_match_with_real_figures(db_session, batch_run_id):
    order = InternalOrder(
        order_id="ORD_AGENT1", gross_amount=Decimal("1000.00"), customer_id="C1",
        payment_method="credit_card", timestamp="2025-01-06T10:00:00+05:30",
    )
    # Deviated MDR (25.00 vs the standard 18.00 for credit_card) -- excluded
    # from the fast path by construction, reaching the discrepancy queue.
    settlement = GatewaySettlement(
        payment_id="PAY_AGENT1", order_id="ORD_AGENT1", gross_amount=Decimal("1000.00"),
        payment_method="credit_card", mdr=Decimal("25.00"), gst_on_mdr=Decimal("4.50"),
        tds_194o=Decimal("1.00"), net_amount=Decimal("969.50"), utr="UTR_AGENT1",
        settlement_date=date_plus_two("2025-01-06"),
    )

    def _stub(record):
        return (
            AgentResolution(
                root_cause_code="AMEX_SURCHARGE", quantified_delta_paise=700,
                evidence_tool_calls=["get_tax_rules"], confidence_note="stub: MDR deviates from standard",
            ),
            {},
        )

    run_batch(db_session, batch_run_id, [order], [settlement], [], diagnose_fn=_stub)

    match = db_session.execute(
        select(ReconciliationMatch).where(
            ReconciliationMatch.batch_run_id == batch_run_id, ReconciliationMatch.order_id == "ORD_AGENT1"
        )
    ).scalar_one()
    assert match.status == "agent_resolved"
    assert "AMEX_SURCHARGE" in match.confidence_note

    rows = db_session.execute(
        select(Account.account_code, JournalLine.direction, JournalLine.amount)
        .join(JournalLine, JournalLine.account_id == Account.account_id)
        .join(JournalEntry, JournalEntry.entry_id == JournalLine.entry_id)
        .where(JournalEntry.batch_run_id == batch_run_id, JournalEntry.reference_id == "ORD_AGENT1")
    ).all()
    mdr_lines = [amount for code, direction, amount in rows if code == MDR_EXPENSE and direction == "D"]
    assert mdr_lines == [Decimal("25.00")], "must post the REAL settlement MDR, not a standard/corrected figure"


# ---------------------------------------------------------------------------
# Test 11 -- orchestrator: short_settlement / duplicate_credit clear
# AR_GATEWAY_CLEARING straight to SUSPENSE_UNRESOLVED for the gross amount,
# with no MDR/GST/TDS recognized (approved design)
# ---------------------------------------------------------------------------

def test_orchestrator_short_settlement_and_duplicate_credit_clear_ar_to_suspense(db_session, batch_run_id):
    # short_settlement: clean rate/timing, zero bank candidates.
    order_a = InternalOrder(
        order_id="ORD_SHORT1", gross_amount=Decimal("500.00"), customer_id="C1",
        payment_method="upi", timestamp="2025-01-06T10:00:00+05:30",
    )
    settlement_a = GatewaySettlement(
        payment_id="PAY_SHORT1", order_id="ORD_SHORT1", gross_amount=Decimal("500.00"),
        payment_method="upi", mdr=Decimal("0.00"), gst_on_mdr=Decimal("0.00"), tds_194o=Decimal("0.50"),
        net_amount=Decimal("499.50"), utr="UTR_SHORT1", settlement_date=date_plus_one("2025-01-06"),
    )

    # duplicate_credit: clean rate/timing, 2 ambiguous bank candidates.
    # UTR is pure alphanumeric (no underscore) so both narrations hit
    # fast_path's phase1 EXACT token regex (UTR[A-Z0-9]+) deterministically,
    # rather than relying on the phase2 fuzzy-fallback score.
    order_b = InternalOrder(
        order_id="ORD_DUP1", gross_amount=Decimal("300.00"), customer_id="C2",
        payment_method="upi", timestamp="2025-01-06T10:00:00+05:30",
    )
    settlement_b = GatewaySettlement(
        payment_id="PAY_DUP1", order_id="ORD_DUP1", gross_amount=Decimal("300.00"),
        payment_method="upi", mdr=Decimal("0.00"), gst_on_mdr=Decimal("0.00"), tds_194o=Decimal("0.30"),
        net_amount=Decimal("299.70"), utr="UTRDUP0001", settlement_date=date_plus_one("2025-01-06"),
    )
    bank_dup1 = BankStatementLine(
        utr="UTRDUP0001", credited_amount=Decimal("299.70"),
        value_date=date_plus_one("2025-01-06"), narration="NEFT-UTRDUP0001-SETTLE",
    )
    bank_dup2 = BankStatementLine(
        utr="UTRDUP0001", credited_amount=Decimal("299.70"),
        value_date=date_plus_one("2025-01-06"), narration="NEFT-UTRDUP0001-SETTLE-DUP",
    )

    run_batch(
        db_session, batch_run_id, [order_a, order_b], [settlement_a, settlement_b], [bank_dup1, bank_dup2],
        diagnose_fn=_stub_diagnose_fn(),
    )

    for order_id, gross_paise in (("ORD_SHORT1", Decimal("500.00")), ("ORD_DUP1", Decimal("300.00"))):
        match = db_session.execute(
            select(ReconciliationMatch).where(
                ReconciliationMatch.batch_run_id == batch_run_id, ReconciliationMatch.order_id == order_id
            )
        ).scalar_one()
        assert match.status == "honest_exception"

        rows = db_session.execute(
            select(Account.account_code, JournalLine.direction, JournalLine.amount)
            .join(JournalLine, JournalLine.account_id == Account.account_id)
            .join(JournalEntry, JournalEntry.entry_id == JournalLine.entry_id)
            .where(JournalEntry.batch_run_id == batch_run_id, JournalEntry.reference_id == order_id)
        ).all()
        codes = {code for code, _d, _a in rows}
        assert MDR_EXPENSE not in codes, f"{order_id}: must not recognize unconfirmed MDR"
        ar_credit = [a for c, d, a in rows if c == AR_GATEWAY_CLEARING and d == "C"]
        suspense_debit = [a for c, d, a in rows if c == SUSPENSE_UNRESOLVED and d == "D"]
        assert ar_credit == [gross_paise]
        assert suspense_debit == [gross_paise]


# ---------------------------------------------------------------------------
# Test 12 -- orchestrator: orphan / adversarial_trap (no order at all) post
# real CASH against SUSPENSE_UNRESOLVED, reference_id UNMATCHED_BANK_<utr>
# ---------------------------------------------------------------------------

def test_orchestrator_orphan_and_adversarial_trap_post_cash_to_suspense(db_session, batch_run_id):
    bank_line = BankStatementLine(
        utr="UTR_ORPHAN1", credited_amount=Decimal("777.00"),
        value_date=date_plus_one("2025-01-06"), narration="NEFT-UTR_ORPHAN1-SETTLE",
    )

    run_batch(db_session, batch_run_id, [], [], [bank_line], diagnose_fn=_stub_diagnose_fn())

    reference_id = "UNMATCHED_BANK_UTR_ORPHAN1"
    match = db_session.execute(
        select(ReconciliationMatch).where(
            ReconciliationMatch.batch_run_id == batch_run_id, ReconciliationMatch.order_id == reference_id
        )
    ).scalar_one()
    assert match.status == "honest_exception"
    assert match.utr == "UTR_ORPHAN1"

    rows = db_session.execute(
        select(Account.account_code, JournalLine.direction, JournalLine.amount)
        .join(JournalLine, JournalLine.account_id == Account.account_id)
        .join(JournalEntry, JournalEntry.entry_id == JournalLine.entry_id)
        .where(JournalEntry.batch_run_id == batch_run_id, JournalEntry.reference_id == reference_id)
    ).all()
    cash_debit = [a for c, d, a in rows if c == CASH and d == "D"]
    suspense_credit = [a for c, d, a in rows if c == SUSPENSE_UNRESOLVED and d == "C"]
    assert cash_debit == [Decimal("777.00")]
    assert suspense_credit == [Decimal("777.00")]


# ---------------------------------------------------------------------------
# Test 13 -- orchestrator: refund_clawback regression -- the new
# category-dispatch logic must still reach the existing tested
# post_refund_clawback_reversal path unchanged
# ---------------------------------------------------------------------------

def test_orchestrator_refund_clawback_regression_unchanged(db_session, batch_run_id):
    order = InternalOrder(
        order_id="ORD_REFUND1", gross_amount=Decimal("1000.00"), customer_id="C1",
        payment_method="credit_card", timestamp="2025-01-06T10:00:00+05:30", refund_amount=Decimal("200.00"),
    )
    settlement = GatewaySettlement(
        payment_id="PAY_REFUND1", order_id="ORD_REFUND1", gross_amount=Decimal("1000.00"),
        payment_method="credit_card", mdr=Decimal("18.00"), gst_on_mdr=Decimal("3.24"),
        tds_194o=Decimal("1.00"), net_amount=Decimal("977.76"), utr="UTR_REFUND1",
        settlement_date=date_plus_two("2025-01-06"),
    )

    def _stub(record):
        return (
            AgentResolution(
                root_cause_code="REFUND_NO_MDR_REVERSAL", quantified_delta_paise=20_000,
                evidence_tool_calls=[], confidence_note="stub: refund present, MDR not reversed",
            ),
            {},
        )

    run_batch(db_session, batch_run_id, [order], [settlement], [], diagnose_fn=_stub)

    match = db_session.execute(
        select(ReconciliationMatch).where(
            ReconciliationMatch.batch_run_id == batch_run_id, ReconciliationMatch.order_id == "ORD_REFUND1"
        )
    ).scalar_one()
    assert match.status == "agent_resolved"

    reversal_key = f"RUN:{batch_run_id}:ORDER:ORD_REFUND1:REFUND_REVERSAL"
    entry = db_session.execute(
        select(JournalEntry).where(JournalEntry.idempotency_key == reversal_key)
    ).scalar_one_or_none()
    assert entry is not None, "post_refund_clawback_reversal must still be invoked"

    rows = db_session.execute(
        select(Account.account_code, JournalLine.direction, JournalLine.amount)
        .join(JournalLine, JournalLine.account_id == Account.account_id)
        .where(JournalLine.entry_id == entry.entry_id)
    ).all()
    codes_directions = {(c, d) for c, d, _a in rows}
    assert (REVENUE_GROSS, "D") in codes_directions
    assert (AR_GATEWAY_CLEARING, "C") in codes_directions
    assert MDR_EXPENSE not in {c for c, _d, _a in rows}, "refund reversal must never touch MDR_EXPENSE"


# ---------------------------------------------------------------------------
# Test 14 -- triggering the frozen batch run makes zero live agent calls
# (reproducibility: the whole suite is network-free)
# ---------------------------------------------------------------------------

def test_trigger_batch_run_frozen_makes_zero_live_agent_calls(client, monkeypatch):
    import src.agent.graph as graph_module

    def _fail_if_called(record):
        raise AssertionError("live diagnose_discrepancy was called for the frozen (fully-cached) batch")

    monkeypatch.setattr(graph_module, "diagnose_discrepancy", _fail_if_called)

    resp = client.post("/batch-runs", json={"source": "frozen"})
    assert resp.status_code == 200


# ---------------------------------------------------------------------------
# _agent_cache_path_for / live-seed cache isolation (src/api/main.py,
# 2026-09-04 addendum): real incident, twice -- a live "seed" trigger
# through the ordinary dashboard/API silently shadowed the frozen dataset's
# permanent cache entries (record_key is just order_id, assigned
# positionally, so two different recipes can share order-id strings). Fixed
# by giving every non-frozen recipe its own cache file, scoped to its exact
# (seed, records) pair.
# ---------------------------------------------------------------------------

def test_agent_cache_path_frozen_always_uses_the_shared_default_path():
    from src.api.main import DEFAULT_AGENT_CACHE_PATH, _agent_cache_path_for

    assert _agent_cache_path_for("frozen", seed=None, records=100) == DEFAULT_AGENT_CACHE_PATH
    assert _agent_cache_path_for("frozen", seed=42, records=100) == DEFAULT_AGENT_CACHE_PATH  # seed ignored for frozen


def test_agent_cache_path_seed_never_matches_the_frozen_path():
    from src.api.main import DEFAULT_AGENT_CACHE_PATH, _agent_cache_path_for

    assert _agent_cache_path_for("seed", seed=1, records=10) != DEFAULT_AGENT_CACHE_PATH


def test_agent_cache_path_seed_scoped_to_its_exact_recipe():
    from src.api.main import _agent_cache_path_for

    same_a = _agent_cache_path_for("seed", seed=1, records=10)
    same_b = _agent_cache_path_for("seed", seed=1, records=10)
    different_seed = _agent_cache_path_for("seed", seed=2, records=10)
    different_records = _agent_cache_path_for("seed", seed=1, records=20)

    assert same_a == same_b, "the identical (seed, records) recipe must reuse its own cache file"
    assert different_seed != same_a
    assert different_records != same_a
    assert different_seed != different_records


def test_live_seed_trigger_never_touches_or_pollutes_the_frozen_cache_file(client, monkeypatch, tmp_path):
    """End-to-end proof through the real HTTP path: a live seed=1, records=10
    trigger (the exact recipe that caused the real 2026-09-04 incident) must
    leave data/agent_runs/layer4_test_cache.jsonl byte-for-byte untouched,
    and must write its own entries to a separate, recipe-scoped file
    instead. Stubs only the actual model call (same pattern as test 14
    above) -- exercises the real default cache-path-selection logic."""
    import src.agent.graph as graph_module
    from src.agent.resolution import AgentResolution
    from src.api.main import AGENT_RUNS_DIR, DEFAULT_AGENT_CACHE_PATH

    def _stub_diagnose(record):
        resolution = AgentResolution(
            root_cause_code="UNRESOLVED", quantified_delta_paise=0, evidence_tool_calls=[], confidence_note="stub"
        )
        return resolution, {"hop_count": 0, "tool_call_history": [], "status": "HONEST_EXCEPTION", "raw_failure": None, "tokens_used": 0}

    monkeypatch.setattr(graph_module, "diagnose_discrepancy", _stub_diagnose)

    frozen_cache_before = DEFAULT_AGENT_CACHE_PATH.read_bytes()
    live_seed_path = AGENT_RUNS_DIR / "live_seed_1_10.jsonl"
    assert not live_seed_path.exists(), "test precondition: no stray file from a prior run"

    try:
        resp = client.post("/batch-runs", json={"source": "seed", "seed": 1, "records": 10})
        assert resp.status_code == 200

        assert DEFAULT_AGENT_CACHE_PATH.read_bytes() == frozen_cache_before, (
            "the frozen dataset's permanent cache must be byte-for-byte untouched by a live seed trigger"
        )
        assert live_seed_path.exists(), "the seed recipe's own cache file must have been created instead"
    finally:
        live_seed_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# batch_run_recipes persistence (src/ledger/models.py, 2026-09-04): the real
# incident this closes -- an in-memory dict lost a run's recipe on every
# server restart, even though its ledger rows stayed in Postgres, orphaning
# the run from every batch_run_id-gated endpoint. Recipe now lives in
# Postgres, not process memory.
# ---------------------------------------------------------------------------

def test_triggered_run_recipe_persists_to_postgres(client, db_session):
    from src.ledger.models import BatchRunRecipe

    resp = client.post("/batch-runs", json={"source": "frozen"})
    assert resp.status_code == 200
    batch_run_id = resp.json()["batch_run_id"]

    row = db_session.get(BatchRunRecipe, uuid.UUID(batch_run_id))
    assert row is not None
    assert row.source == "frozen"
    assert row.records == 100


def test_status_endpoint_depends_only_on_the_persisted_recipe_row_not_any_process_state(client, db_session):
    """Simulates exactly what the real incident looked like, both
    directions: (1) real ledger data with its recipe row gone (the old
    bug's symptom -- proves there's no leftover in-memory fallback quietly
    covering for it) correctly 404s; (2) re-inserting only the recipe row
    for that same already-real data makes it recoverable again, with zero
    re-triggering and zero re-posting -- proving the fix is a genuine
    Postgres-backed lookup, not something scoped to the triggering request."""
    from src.ledger.models import BatchRunRecipe

    resp = client.post("/batch-runs", json={"source": "frozen"})
    assert resp.status_code == 200
    batch_run_id = resp.json()["batch_run_id"]

    assert client.get(f"/batch-runs/{batch_run_id}/status").status_code == 200

    row = db_session.get(BatchRunRecipe, uuid.UUID(batch_run_id))
    db_session.delete(row)
    db_session.commit()

    resp = client.get(f"/batch-runs/{batch_run_id}/status")
    assert resp.status_code == 404, "with the recipe row gone, this must 404 -- no hidden in-memory fallback"

    db_session.add(BatchRunRecipe(batch_run_id=uuid.UUID(batch_run_id), source="frozen", seed=None, records=100))
    db_session.commit()

    resp = client.get(f"/batch-runs/{batch_run_id}/status")
    assert resp.status_code == 200, "re-inserting the recipe row alone must recover access to the still-real ledger data"
    assert resp.json()["total"] > 0


# ---------------------------------------------------------------------------
# Test 15 -- the seed/live source path is wired through an injectable
# diagnose_fn -- no real Groq call is required to exercise it in this suite
# ---------------------------------------------------------------------------

def test_trigger_batch_run_seed_source_uses_injected_diagnose_fn_no_network(client):
    from src.api.main import app, get_diagnose_fn

    app.dependency_overrides[get_diagnose_fn] = _stub_diagnose_fn
    try:
        resp = client.post("/batch-runs", json={"source": "seed", "seed": 999, "records": 10})
    finally:
        del app.dependency_overrides[get_diagnose_fn]

    assert resp.status_code == 200
    body = resp.json()
    assert body["total_orders"] > 0


# ---------------------------------------------------------------------------
# AgentRateLimitedError surfaces as a clean 503 with a real detail message,
# not FastAPI's default opaque 500 (2026-09-03 debugging session: a live
# groq.RateLimitError previously propagated uncaught all the way here). No
# live model call needed -- the injected diagnose_fn raises the same typed
# exception the real live path raises once daily_token_tracker.check_budget()
# or _invoke_with_backoff (src/agent/graph.py) detects a daily-quota 429.
# ---------------------------------------------------------------------------

def test_trigger_batch_run_surfaces_clean_503_on_agent_rate_limit(client):
    from src.agent.rate_limiter import AgentRateLimitedError
    from src.api.main import app, get_diagnose_fn

    def _rate_limited_diagnose_fn(record):
        raise AgentRateLimitedError("Local daily token budget nearly exhausted: 199000/200000 used today")

    app.dependency_overrides[get_diagnose_fn] = lambda: _rate_limited_diagnose_fn
    try:
        resp = client.post("/batch-runs", json={"source": "seed", "seed": 999, "records": 10})
    finally:
        del app.dependency_overrides[get_diagnose_fn]

    assert resp.status_code == 503
    assert "daily token budget" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# run_batch's pre-flight budget check (src/orchestration/batch_runner.py
# addendum, 2026-09-03): exercises the REAL default diagnose_fn path (no
# dependency override) against records guaranteed uncached at an isolated
# tmp_path cache file, with the real daily_token_tracker singleton
# pre-loaded near-exhaustion to force the check to fire. Proves the whole
# point of moving this check before Stage 1: zero rows written anywhere.
# ---------------------------------------------------------------------------

def test_run_batch_preflight_blocks_before_any_posting_when_budget_insufficient(db_session, tmp_path):
    from src.agent.rate_limiter import DAILY_TOKEN_BUDGET, AgentRateLimitedError, daily_token_tracker
    from src.data.generator import generate_batch

    batch_run_id = uuid.uuid4()
    batch = generate_batch(num_records=10, seed=424242)  # verified: 3 records need diagnosis, none ever cached
    empty_cache = tmp_path / "empty_cache.jsonl"

    daily_token_tracker.reset_for_testing()
    daily_token_tracker.record_usage(DAILY_TOKEN_BUDGET)  # simulate a fully exhausted day
    try:
        with pytest.raises(AgentRateLimitedError, match="estimated"):
            run_batch(
                db_session, batch_run_id, batch.orders, batch.settlements, batch.bank_lines,
                agent_cache_path=empty_cache,
            )
    finally:
        daily_token_tracker.reset_for_testing()

    assert not empty_cache.exists(), "no live call (and therefore no cache write) should ever have been attempted"
    entries = db_session.execute(
        select(JournalEntry).where(JournalEntry.batch_run_id == batch_run_id)
    ).scalars().all()
    matches = db_session.execute(
        select(ReconciliationMatch).where(ReconciliationMatch.batch_run_id == batch_run_id)
    ).scalars().all()
    assert entries == [], "Stage 1 capture must not have posted anything -- the check runs before it now"
    assert matches == []


def test_run_batch_preflight_ignores_as_of_gated_records(db_session, tmp_path):
    """A record whose settlement/bank-credit postdates as_of never reaches
    diagnose_fn at all (the as_of addendum above) -- it must not count
    toward the pre-flight estimate either, or an as_of-limited run could be
    blocked by budget it will never actually spend."""
    from datetime import date as _date

    from src.agent.rate_limiter import DAILY_TOKEN_BUDGET, daily_token_tracker
    from src.data.generator import generate_batch

    batch_run_id = uuid.uuid4()
    batch = generate_batch(num_records=10, seed=424242)
    empty_cache = tmp_path / "empty_cache.jsonl"

    daily_token_tracker.reset_for_testing()
    daily_token_tracker.record_usage(DAILY_TOKEN_BUDGET)  # exhausted -- would block if anything were counted
    try:
        # as_of before the dataset's own window (2025-01-06 onward) gates every
        # settlement/bank-credit out, so 0 records actually need diagnosis --
        # must proceed (posting only Stage 1 captures), not raise.
        summary = run_batch(
            db_session, batch_run_id, batch.orders, batch.settlements, batch.bank_lines,
            agent_cache_path=empty_cache, as_of=_date(2020, 1, 1),
        )
    finally:
        daily_token_tracker.reset_for_testing()

    assert summary.total_orders == len(batch.orders)


# ---------------------------------------------------------------------------
# Tests 16-20 -- as_of-gated Stage 2 posting (Layer 6 addendum, approved
# after Layer 7: see src/orchestration/batch_runner.py's module docstring
# for why this exists -- without it, project_cashflow()'s "projected"
# status was structurally unreachable through any real triggered run).
# ---------------------------------------------------------------------------

def test_orchestrator_as_of_gates_fast_path_stage_2_for_future_settlement(db_session, batch_run_id):
    from datetime import date as _date

    # UTR is pure alphanumeric (no underscore) so the narration hits
    # fast_path's phase1 EXACT token regex (UTR[A-Z0-9]+) deterministically,
    # same convention as test 11 above -- this test needs a genuine
    # fast-path clean match to exercise the fast_path-loop as_of gate
    # specifically, not just fall through to the discrepancy queue.
    order = InternalOrder(
        order_id="ORD_FUTURE1", gross_amount=Decimal("1000.00"), customer_id="C1",
        payment_method="upi", timestamp="2025-01-06T10:00:00+05:30",
    )
    # UPI is T+1: order captured 2025-01-06 settles 2025-01-07 for a genuine
    # fast-path clean match (verified: same-day and T+2 both fail matching
    # and fall to the discrepancy queue instead -- only exact T+1 resolves).
    settlement = GatewaySettlement(
        payment_id="PAY_FUTURE1", order_id="ORD_FUTURE1", gross_amount=Decimal("1000.00"),
        payment_method="upi", mdr=Decimal("0.00"), gst_on_mdr=Decimal("0.00"), tds_194o=Decimal("1.00"),
        net_amount=Decimal("999.00"), utr="UTRFUTURE0001", settlement_date=_date(2025, 1, 7),
    )
    bank_line = BankStatementLine(
        utr="UTRFUTURE0001", credited_amount=Decimal("999.00"),
        value_date=_date(2025, 1, 7), narration="NEFT-UTRFUTURE0001-SETTLE",
    )

    summary = run_batch(
        db_session, batch_run_id, [order], [settlement], [bank_line],
        diagnose_fn=_stub_diagnose_fn(), as_of=_date(2025, 1, 6),
    )
    assert summary.fast_path_count == 0  # gated -- not counted as resolved this run

    rows = db_session.execute(
        select(Account.account_code, JournalLine.direction, JournalLine.amount)
        .join(JournalLine, JournalLine.account_id == Account.account_id)
        .join(JournalEntry, JournalEntry.entry_id == JournalLine.entry_id)
        .where(JournalEntry.batch_run_id == batch_run_id, JournalEntry.reference_id == "ORD_FUTURE1")
    ).all()
    assert (REVENUE_GROSS, "C", Decimal("1000.00")) in rows, "Stage 1 (capture) must still post regardless of as_of"
    assert CASH not in {c for c, _d, _a in rows}, "Stage 2 must not post for a settlement dated after as_of"

    match = db_session.execute(
        select(ReconciliationMatch).where(
            ReconciliationMatch.batch_run_id == batch_run_id, ReconciliationMatch.order_id == "ORD_FUTURE1"
        )
    ).scalar_one_or_none()
    assert match is None, "an order not yet reconciled as of this date must carry no reconciliation_matches row"


def test_orchestrator_as_of_boundary_is_inclusive(db_session, batch_run_id):
    from datetime import date as _date

    order = InternalOrder(
        order_id="ORD_ONTIME1", gross_amount=Decimal("500.00"), customer_id="C1",
        payment_method="upi", timestamp="2025-01-06T10:00:00+05:30",
    )
    # UPI T+1: order captured 2025-01-06 settles 2025-01-07 -- as_of is set
    # to that same 2025-01-07 to prove the boundary (`>`, not `>=`) posts
    # normally rather than gating.
    settlement = GatewaySettlement(
        payment_id="PAY_ONTIME1", order_id="ORD_ONTIME1", gross_amount=Decimal("500.00"),
        payment_method="upi", mdr=Decimal("0.00"), gst_on_mdr=Decimal("0.00"), tds_194o=Decimal("0.50"),
        net_amount=Decimal("499.50"), utr="UTRONTIME0001", settlement_date=_date(2025, 1, 7),
    )
    bank_line = BankStatementLine(
        utr="UTRONTIME0001", credited_amount=Decimal("499.50"),
        value_date=_date(2025, 1, 7), narration="NEFT-UTRONTIME0001-SETTLE",
    )

    summary = run_batch(
        db_session, batch_run_id, [order], [settlement], [bank_line],
        diagnose_fn=_stub_diagnose_fn(), as_of=_date(2025, 1, 7),
    )
    assert summary.fast_path_count == 1, "a settlement dated exactly on as_of must still post normally"

    rows = db_session.execute(
        select(Account.account_code, JournalLine.direction, JournalLine.amount)
        .join(JournalLine, JournalLine.account_id == Account.account_id)
        .join(JournalEntry, JournalEntry.entry_id == JournalLine.entry_id)
        .where(JournalEntry.batch_run_id == batch_run_id, JournalEntry.reference_id == "ORD_ONTIME1")
    ).all()
    assert (CASH, "D", Decimal("499.50")) in rows


def test_orchestrator_as_of_gates_discrepancy_queue_and_skips_diagnosis(db_session, batch_run_id):
    from datetime import date as _date

    order = InternalOrder(
        order_id="ORD_FUTURE2", gross_amount=Decimal("1000.00"), customer_id="C1",
        payment_method="credit_card", timestamp="2025-01-06T10:00:00+05:30",
    )
    # Deviated MDR -- excluded from the fast path, reaching the discrepancy queue.
    settlement = GatewaySettlement(
        payment_id="PAY_FUTURE2", order_id="ORD_FUTURE2", gross_amount=Decimal("1000.00"),
        payment_method="credit_card", mdr=Decimal("25.00"), gst_on_mdr=Decimal("4.50"),
        tds_194o=Decimal("1.00"), net_amount=Decimal("969.50"), utr="UTR_FUTURE2",
        settlement_date=_date(2025, 1, 10),
    )

    calls = []

    def _spy(record):
        calls.append(record)
        return (
            AgentResolution(
                root_cause_code="AMEX_SURCHARGE", quantified_delta_paise=700,
                evidence_tool_calls=[], confidence_note="stub",
            ),
            {},
        )

    summary = run_batch(
        db_session, batch_run_id, [order], [settlement], [],
        diagnose_fn=_spy, as_of=_date(2025, 1, 6),
    )
    assert summary.agent_resolved_count == 0
    assert calls == [], "diagnose_fn must not be called for a settlement dated after as_of"

    match = db_session.execute(
        select(ReconciliationMatch).where(
            ReconciliationMatch.batch_run_id == batch_run_id, ReconciliationMatch.order_id == "ORD_FUTURE2"
        )
    ).scalar_one_or_none()
    assert match is None


def test_orchestrator_as_of_gates_unmatched_bank_line_and_skips_diagnosis(db_session, batch_run_id):
    from datetime import date as _date

    bank_line = BankStatementLine(
        utr="UTR_FUTURE_ORPHAN", credited_amount=Decimal("777.00"),
        value_date=_date(2025, 1, 10), narration="NEFT-UTR_FUTURE_ORPHAN-SETTLE",
    )

    calls = []

    def _spy(record):
        calls.append(record)
        return _stub_diagnose_fn()(record)

    summary = run_batch(
        db_session, batch_run_id, [], [], [bank_line],
        diagnose_fn=_spy, as_of=_date(2025, 1, 6),
    )
    assert summary.honest_exception_count == 0
    assert calls == [], "diagnose_fn must not be called for a bank credit dated after as_of"

    reference_id = "UNMATCHED_BANK_UTR_FUTURE_ORPHAN"
    entry = db_session.execute(
        select(JournalEntry).where(
            JournalEntry.batch_run_id == batch_run_id, JournalEntry.reference_id == reference_id
        )
    ).scalar_one_or_none()
    assert entry is None, "a bank credit not yet reconciled as of this date must carry no journal entry"


def test_as_of_gated_frozen_trigger_produces_a_real_projected_forecast_row(client, db_session):
    """End-to-end proof, through the real HTTP path, that this fix actually
    achieves its goal: a frozen batch triggered with an as_of cutoff that
    splits its settlement dates leaves some orders genuinely projected, and
    project_cashflow() -- called with that same as_of -- reports it as such.
    Before this fix, "projected" was unreachable via any real triggered run
    (see the Layer 7 dashboard follow-up conversation)."""
    cutoff = "2025-01-20"  # splits the frozen dataset's settlement dates 41/46

    resp = client.post("/batch-runs", json={"source": "frozen", "as_of": cutoff})
    assert resp.status_code == 200
    body = resp.json()
    batch_run_id = body["batch_run_id"]

    # Fewer orders got fully reconciled this run than the unconditional total.
    assert body["fast_path_count"] + body["agent_resolved_count"] < FROZEN_FAST_PATH_COUNT + FROZEN_AGENT_RESOLVED_COUNT

    forecast_resp = client.get(f"/batch-runs/{batch_run_id}/forecast", params={"as_of": cutoff, "horizon_days": 7})
    assert forecast_resp.status_code == 200
    forecast_rows = forecast_resp.json()
    statuses = {row["account_status"] for row in forecast_rows}
    assert "projected" in statuses, "a real as_of-gated trigger must produce at least one genuinely projected row"
    assert "confirmed" in statuses


# ---------------------------------------------------------------------------
# Small date helpers -- avoid importing datetime at module scope repeatedly
# ---------------------------------------------------------------------------

def date_plus_one(iso: str):
    from datetime import date, timedelta

    y, m, d = map(int, iso.split("-"))
    return date(y, m, d) + timedelta(days=1)


def date_plus_two(iso: str):
    from datetime import date, timedelta

    y, m, d = map(int, iso.split("-"))
    return date(y, m, d) + timedelta(days=2)


# ---------------------------------------------------------------------------
# _should_apply_schema_on_startup -- the guard that keeps the app's
# startup schema-check (lifespan, wired around ensure_schema_exists) from
# ever running against the real finance_controller database during a test
# run. Every fixture above overrides get_session before constructing
# TestClient(app), which is exactly the condition this checks; a direct,
# synchronous unit test of the guard itself, not the async lifespan
# wrapper -- see src/api/main.py's lifespan docstring for why.
# ---------------------------------------------------------------------------

def test_should_apply_schema_on_startup_true_when_get_session_is_not_overridden():
    from src.api.main import _should_apply_schema_on_startup, app, get_session

    app.dependency_overrides.pop(get_session, None)
    assert _should_apply_schema_on_startup(app) is True


def test_should_apply_schema_on_startup_false_when_get_session_is_overridden():
    from src.api.main import _should_apply_schema_on_startup, app, get_session

    app.dependency_overrides[get_session] = lambda: None
    try:
        assert _should_apply_schema_on_startup(app) is False
    finally:
        app.dependency_overrides.pop(get_session, None)
