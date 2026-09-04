"""SQLAlchemy models and connection/schema helpers for the Layer 3 ledger.

The schema itself is authored once, in `db_schema.sql` (ENUM types, the
deferred constraint trigger, the seed chart of accounts) -- these ORM classes
map onto that schema for querying and inserting, they do not redefine it via
`Base.metadata.create_all()`. `create_schema()` / `drop_schema()` execute
`db_schema.sql` (and its inverse) as raw SQL against a real Postgres database;
there is no non-Postgres code path, since the trigger and ENUM types are not
portable and CLAUDE.md forbids mocking this layer.
"""
from __future__ import annotations

import os
import uuid
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from sqlalchemy import (
    CHAR,
    ForeignKey,
    Numeric,
    String,
    TIMESTAMP,
    Text,
    UniqueConstraint,
    create_engine,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import ENUM as PG_ENUM
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker

SCHEMA_SQL_PATH = Path(__file__).parent / "db_schema.sql"

EntryStatus = PG_ENUM("posted", "reversed", name="entry_status", create_type=False)
AccountType = PG_ENUM(
    "asset", "liability", "revenue", "expense", "suspense", name="account_type", create_type=False
)
MatchStatus = PG_ENUM(
    "fast_path", "agent_resolved", "honest_exception", name="match_status", create_type=False
)


class Base(DeclarativeBase):
    pass


class Account(Base):
    __tablename__ = "accounts"

    account_id: Mapped[int] = mapped_column(primary_key=True)
    account_code: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    account_name: Mapped[str] = mapped_column(String(200), nullable=False)
    account_type: Mapped[str] = mapped_column(AccountType, nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())


class JournalEntry(Base):
    __tablename__ = "journal_entries"

    entry_id: Mapped[int] = mapped_column(primary_key=True)
    batch_run_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    idempotency_key: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    reference_id: Mapped[str] = mapped_column(String(100), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(EntryStatus, nullable=False, server_default="posted")
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

    lines: Mapped[list["JournalLine"]] = relationship(back_populates="entry")


class JournalLine(Base):
    __tablename__ = "journal_lines"

    line_id: Mapped[int] = mapped_column(primary_key=True)
    entry_id: Mapped[int] = mapped_column(ForeignKey("journal_entries.entry_id"), nullable=False)
    account_id: Mapped[int] = mapped_column(ForeignKey("accounts.account_id"), nullable=False)
    direction: Mapped[str] = mapped_column(CHAR(1), nullable=False)
    amount: Mapped[Decimal] = mapped_column(Numeric(14, 2), nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())

    entry: Mapped["JournalEntry"] = relationship(back_populates="lines")


class ReconciliationMatch(Base):
    __tablename__ = "reconciliation_matches"
    __table_args__ = (UniqueConstraint("batch_run_id", "order_id"),)

    match_id: Mapped[int] = mapped_column(primary_key=True)
    batch_run_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), nullable=False)
    order_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    payment_id: Mapped[str | None] = mapped_column(String(100), nullable=True)
    utr: Mapped[str | None] = mapped_column(String(100), nullable=True)
    status: Mapped[str] = mapped_column(MatchStatus, nullable=False)
    confidence_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    journal_entry_id: Mapped[int | None] = mapped_column(
        ForeignKey("journal_entries.entry_id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())


class BatchRunRecipe(Base):
    """Persists which (source, seed, records) recipe produced a given
    batch_run_id -- replaces src/api/main.py's old in-process dict, which
    lost this mapping on every server restart even though the ledger rows
    it described were safely in Postgres the whole time (2026-09-04
    incident)."""

    __tablename__ = "batch_run_recipes"

    batch_run_id: Mapped[uuid.UUID] = mapped_column(PG_UUID(as_uuid=True), primary_key=True)
    source: Mapped[str] = mapped_column(String(20), nullable=False)
    seed: Mapped[int | None] = mapped_column(nullable=True)
    records: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(TIMESTAMP(timezone=True), server_default=func.now())


def get_database_url() -> str:
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg://postgres:postgres@localhost:5433/finance_controller",
    )


def get_engine(database_url: str | None = None):
    return create_engine(database_url or get_database_url(), future=True)


def get_sessionmaker(engine=None):
    return sessionmaker(bind=engine or get_engine(), future=True, expire_on_commit=False)


_DROP_SCHEMA_SQL = """
DROP TABLE IF EXISTS reconciliation_matches CASCADE;
DROP TABLE IF EXISTS journal_lines CASCADE;
DROP TABLE IF EXISTS journal_entries CASCADE;
DROP TABLE IF EXISTS accounts CASCADE;
DROP FUNCTION IF EXISTS check_entry_balances() CASCADE;
DROP TYPE IF EXISTS match_status;
DROP TYPE IF EXISTS account_type;
DROP TYPE IF EXISTS entry_status;
"""


def drop_schema(engine) -> None:
    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        cur.execute(_DROP_SCHEMA_SQL)
        raw.commit()
    finally:
        raw.close()


def create_schema(engine) -> None:
    sql = SCHEMA_SQL_PATH.read_text(encoding="utf-8")
    raw = engine.raw_connection()
    try:
        cur = raw.cursor()
        cur.execute(sql)
        raw.commit()
    finally:
        raw.close()


def reset_schema(engine) -> None:
    """Drop and recreate the ledger schema fresh. Test-only convenience --
    never wired to anything the demo path depends on (CLAUDE.md Sec.3.3:
    the app itself is batch_run_id-scoped, not destructively reset)."""
    drop_schema(engine)
    create_schema(engine)


_BATCH_RUN_RECIPES_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS batch_run_recipes (
    batch_run_id UUID PRIMARY KEY,
    source VARCHAR(20) NOT NULL,
    seed INT,
    records INT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def ensure_batch_run_recipes_table_exists(engine) -> None:
    """Creates batch_run_recipes if it's missing, independent of whether the
    REST of the schema is already present -- this table was added after
    db_schema.sql's other tables, so an existing, already-schema'd database
    (accounts already exists -> ensure_schema_exists's own check is a no-op)
    would otherwise never get it. IF NOT EXISTS makes this safe to call
    unconditionally, every startup, alongside ensure_schema_exists."""
    with engine.connect() as conn:
        conn.execute(text(_BATCH_RUN_RECIPES_TABLE_SQL))
        conn.commit()


def ensure_schema_exists(engine) -> None:
    """Applies db_schema.sql if the ledger schema isn't already present on
    this database -- idempotent, safe to call every time the app starts.
    db_schema.sql itself is plain `CREATE TYPE`/`CREATE TABLE` (no `IF NOT
    EXISTS`), so it errors on a re-run against an already-schema'd database;
    this checks first via `to_regclass`, a cheap, standard Postgres
    existence probe, and only calls create_schema() when the `accounts`
    table is genuinely absent. Unlike reset_schema (test-only, destructive),
    this never drops anything -- it exists to bridge the gap between a
    freshly (re)created Postgres container (e.g. after `docker compose down
    -v`) and the schema the app expects, without a manual
    `psql -f db_schema.sql` step. Wired into src/api/main.py's startup."""
    with engine.connect() as conn:
        already_exists = conn.execute(text("SELECT to_regclass('public.accounts')")).scalar()
    if already_exists is None:
        create_schema(engine)
    ensure_batch_run_recipes_table_exists(engine)


def ensure_database_exists(admin_database_url: str, target_db_name: str) -> None:
    """Create `target_db_name` on the server addressed by `admin_database_url`
    if it does not already exist. Used by the test fixture to provision a
    dedicated test database without touching the app's own database."""
    admin_engine = create_engine(admin_database_url, future=True, isolation_level="AUTOCOMMIT")
    try:
        with admin_engine.connect() as conn:
            exists = conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": target_db_name}
            ).scalar()
            if not exists:
                conn.execute(text(f'CREATE DATABASE "{target_db_name}"'))
    finally:
        admin_engine.dispose()
