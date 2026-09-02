"""Layer 6: thin FastAPI transport over src/ledger, src/matching,
src/forecast, and src/orchestration/batch_runner.py (docs/plan.md Layer 6).
Every number this API serves traces to an existing tested function in those
modules -- the only code added here is HTTP routing, request/response
shaping, and an in-memory batch_run_id -> recipe registry.

Registry design note: src/data/generator.py's determinism is this project's
central claim (CLAUDE.md Sec.5) -- generate_batch(records, seed) is
byte-identical every time. So a "seed" batch run's raw order/settlement/bank
data never needs to be persisted to be re-read later; the registry only
remembers the RECIPE (source/seed/records) per batch_run_id, not the data
itself, and regenerates on demand. It lives in process memory (a single
FastAPI process, consistent with CLAUDE.md Sec.2's modular-monolith framing)
and is lost on restart -- an accepted, documented tradeoff: the ledger rows
themselves stay in Postgres, keyed by batch_run_id, and are still directly
queryable there even if the registry forgets which recipe produced them.
"""
from __future__ import annotations

import json
import uuid
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, model_validator
from sqlalchemy import func as sa_func
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.data.generator import generate_batch
from src.data.models import BankStatementLine, GatewaySettlement, InternalOrder
from src.forecast.cashflow import project_cashflow
from src.ledger.journal import trial_balance
from src.ledger.models import ReconciliationMatch, ensure_schema_exists, get_engine, get_sessionmaker
from src.orchestration.batch_runner import DiagnoseFn, run_batch

FROZEN_DIR = Path(__file__).resolve().parents[2] / "data" / "challenge_batch_100"

_engine = None
_session_factory = None


def _get_session_factory():
    global _engine, _session_factory
    if _session_factory is None:
        _engine = get_engine()
        _session_factory = get_sessionmaker(_engine)
    return _session_factory


def get_session():
    session = _get_session_factory()()
    try:
        yield session
    finally:
        session.close()


def _should_apply_schema_on_startup(app: FastAPI) -> bool:
    """True unless get_session has been overridden -- i.e. running under the
    test suite's own fixtures, which apply the ledger schema to a separate
    finance_controller_test database via tests/conftest.py's pg_engine
    fixture (see conftest.py's module docstring). Startup schema-application
    must never also touch the real finance_controller database from a test
    run, so it's skipped whenever a test has already swapped get_session."""
    return get_session not in app.dependency_overrides


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Applies db_schema.sql on startup if the ledger schema isn't already
    present -- without this, a freshly (re)created Postgres container (e.g.
    after `docker compose down -v`) serves a 500 on every route until
    someone manually pipes db_schema.sql into psql. See
    ensure_schema_exists's docstring in src/ledger/models.py."""
    if _should_apply_schema_on_startup(app):
        _get_session_factory()
        ensure_schema_exists(_engine)
    yield


app = FastAPI(title="ZeroDrift API", lifespan=lifespan)


def get_diagnose_fn() -> DiagnoseFn | None:
    """Overridable seam, mirroring src/agent/graph.py's own testability
    pattern: returns None by default, meaning run_batch falls back to its
    own cache/replay-backed default (zero live model calls against the
    frozen dataset, which is fully cached -- see batch_runner.py). Tests
    override this dependency to inject a stub for the "seed" source path,
    which has no pre-existing cache and would otherwise require a live
    model call to exercise."""
    return None


# batch_run_id -> {"source", "seed", "records"}. See module docstring.
_BATCH_RUN_REGISTRY: dict[uuid.UUID, dict] = {}


def _load_frozen() -> tuple[list[InternalOrder], list[GatewaySettlement], list[BankStatementLine]]:
    orders = [
        InternalOrder.model_validate(d)
        for d in json.loads((FROZEN_DIR / "internal_orders.json").read_text(encoding="utf-8"))
    ]
    settlements = [
        GatewaySettlement.model_validate(d)
        for d in json.loads((FROZEN_DIR / "gateway_settlement.json").read_text(encoding="utf-8"))
    ]
    bank_lines = [
        BankStatementLine.model_validate(d)
        for d in json.loads((FROZEN_DIR / "bank_statement.json").read_text(encoding="utf-8"))
    ]
    return orders, settlements, bank_lines


def _load_for_recipe(recipe: dict) -> tuple[list[InternalOrder], list[GatewaySettlement], list[BankStatementLine]]:
    if recipe["source"] == "frozen":
        return _load_frozen()
    batch = generate_batch(num_records=recipe["records"], seed=recipe["seed"])
    return batch.orders, batch.settlements, batch.bank_lines


class TriggerBatchRunRequest(BaseModel):
    source: Literal["frozen", "seed"] = "frozen"
    seed: int | None = None
    records: int = 100
    as_of: date | None = None

    @model_validator(mode="after")
    def seed_required_for_seed_source(self) -> "TriggerBatchRunRequest":
        if self.source == "seed" and self.seed is None:
            raise ValueError("seed is required when source='seed'")
        return self


class BatchRunSummaryResponse(BaseModel):
    batch_run_id: str
    total_orders: int
    total_unmatched_bank_lines: int
    fast_path_count: int
    agent_resolved_count: int
    honest_exception_count: int


class ReconciliationStatusResponse(BaseModel):
    batch_run_id: str
    fast_path: int
    agent_resolved: int
    honest_exception: int
    total: int


class ExceptionRecord(BaseModel):
    order_id: str
    utr: str | None
    status: str
    confidence_note: str | None


class TrialBalanceRow(BaseModel):
    account_code: str
    account_name: str
    account_type: str
    debit_total_paise: int
    credit_total_paise: int
    net_balance_paise: int


class ForecastRow(BaseModel):
    order_id: str
    payment_method: str
    is_international: bool
    account_status: str
    expected_cash_date: str | None
    within_horizon: bool
    amount_paise: int
    low_paise: int
    high_paise: int


def _require_known_run(batch_run_id: uuid.UUID) -> dict:
    recipe = _BATCH_RUN_REGISTRY.get(batch_run_id)
    if recipe is None:
        raise HTTPException(status_code=404, detail=f"unknown batch_run_id: {batch_run_id}")
    return recipe


@app.post("/batch-runs", response_model=BatchRunSummaryResponse)
def trigger_batch_run(
    req: TriggerBatchRunRequest,
    session: Session = Depends(get_session),
    diagnose_fn: DiagnoseFn | None = Depends(get_diagnose_fn),
) -> BatchRunSummaryResponse:
    batch_run_id = uuid.uuid4()
    orders, settlements, bank_lines = _load_for_recipe(req.model_dump())
    summary = run_batch(
        session, batch_run_id, orders, settlements, bank_lines, diagnose_fn=diagnose_fn, as_of=req.as_of
    )
    _BATCH_RUN_REGISTRY[batch_run_id] = {"source": req.source, "seed": req.seed, "records": req.records}
    return BatchRunSummaryResponse(
        batch_run_id=str(batch_run_id),
        total_orders=summary.total_orders,
        total_unmatched_bank_lines=summary.total_unmatched_bank_lines,
        fast_path_count=summary.fast_path_count,
        agent_resolved_count=summary.agent_resolved_count,
        honest_exception_count=summary.honest_exception_count,
    )


@app.get("/batch-runs/{batch_run_id}/status", response_model=ReconciliationStatusResponse)
def get_status(batch_run_id: uuid.UUID, session: Session = Depends(get_session)) -> ReconciliationStatusResponse:
    _require_known_run(batch_run_id)
    rows = session.execute(
        select(ReconciliationMatch.status, sa_func.count())
        .where(ReconciliationMatch.batch_run_id == batch_run_id)
        .group_by(ReconciliationMatch.status)
    ).all()
    counts = {status: count for status, count in rows}
    return ReconciliationStatusResponse(
        batch_run_id=str(batch_run_id),
        fast_path=counts.get("fast_path", 0),
        agent_resolved=counts.get("agent_resolved", 0),
        honest_exception=counts.get("honest_exception", 0),
        total=sum(counts.values()),
    )


@app.get("/batch-runs/{batch_run_id}/exceptions", response_model=list[ExceptionRecord])
def get_exceptions(batch_run_id: uuid.UUID, session: Session = Depends(get_session)) -> list[ExceptionRecord]:
    _require_known_run(batch_run_id)
    rows = (
        session.execute(
            select(ReconciliationMatch)
            .where(ReconciliationMatch.batch_run_id == batch_run_id, ReconciliationMatch.status == "honest_exception")
            .order_by(ReconciliationMatch.order_id)
        )
        .scalars()
        .all()
    )
    return [
        ExceptionRecord(order_id=r.order_id, utr=r.utr, status=r.status, confidence_note=r.confidence_note)
        for r in rows
    ]


@app.get("/batch-runs/{batch_run_id}/trial-balance", response_model=list[TrialBalanceRow])
def get_trial_balance(batch_run_id: uuid.UUID, session: Session = Depends(get_session)) -> list[TrialBalanceRow]:
    _require_known_run(batch_run_id)
    tb = trial_balance(session, batch_run_id)
    return [TrialBalanceRow(**row) for row in tb.to_dicts()]


@app.get("/batch-runs/{batch_run_id}/forecast", response_model=list[ForecastRow])
def get_forecast(
    batch_run_id: uuid.UUID,
    as_of: date,
    horizon_days: int = 7,
    session: Session = Depends(get_session),
) -> list[ForecastRow]:
    recipe = _require_known_run(batch_run_id)
    orders, settlements, _bank_lines = _load_for_recipe(recipe)
    result = project_cashflow(session, batch_run_id, orders, settlements, as_of=as_of, horizon_days=horizon_days)
    return [
        ForecastRow(
            order_id=row["order_id"],
            payment_method=row["payment_method"],
            is_international=row["is_international"],
            account_status=row["account_status"],
            expected_cash_date=row["expected_cash_date"].isoformat() if row["expected_cash_date"] else None,
            within_horizon=row["within_horizon"],
            amount_paise=row["amount_paise"],
            low_paise=row["low_paise"],
            high_paise=row["high_paise"],
        )
        for row in result.to_dicts()
    ]
