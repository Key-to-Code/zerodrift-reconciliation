"""Shared fixtures for tests that need a real Postgres database (Layer 3).

CLAUDE.md forbids mocking the ledger layer -- the ENUM types and the
deferred constraint trigger are Postgres-specific and not something a mock
or an in-memory substitute could honestly exercise. These fixtures point at
a dedicated `finance_controller_test` database on the docker-compose
Postgres instance (see docker-compose.yml, host port 5433), created fresh
each test session via `reset_schema` -- never against the app's own
`finance_controller` database.
"""
import os
import uuid

import pytest
from sqlalchemy import create_engine

from src.ledger.models import ensure_database_exists, get_sessionmaker, reset_schema

TEST_DB_NAME = "finance_controller_test"
ADMIN_DATABASE_URL = os.environ.get(
    "ADMIN_DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5433/postgres"
)
TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", f"postgresql+psycopg://postgres:postgres@localhost:5433/{TEST_DB_NAME}"
)


@pytest.fixture(scope="session")
def pg_engine():
    ensure_database_exists(ADMIN_DATABASE_URL, TEST_DB_NAME)
    engine = create_engine(TEST_DATABASE_URL, future=True)
    reset_schema(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session(pg_engine):
    session_factory = get_sessionmaker(pg_engine)
    session = session_factory()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture()
def batch_run_id() -> uuid.UUID:
    return uuid.uuid4()
