"""Persistent cache/replay for live agent invocations.

docs/plan.md specifies data/agent_runs/<seed>_<run_index>.jsonl logging and
an evaluate.py --replay mode as Layer 8 concerns. Built here, in Layer 4,
instead of deferred -- motivated directly by hitting Groq's free-tier daily
token cap mid-layer (200,000 TPD, ~4,400 tokens/record-run observed here ->
roughly 45 record-runs/day on this key). Re-running the full live suite
from scratch every time a tool/prompt change needs validating is not
sustainable at that budget. Once a record has been successfully diagnosed
against the CURRENT agent logic, the result is cached permanently; a later
run reads it back with zero API calls instead of re-asking the model.

Cache invalidation is tied to AGENT_LOGIC_VERSION (src/agent/graph.py), not
just the record's own content hash -- a change to SYSTEM_PROMPT,
FINAL_ANSWER_INSTRUCTION, a tool's behavior, or the model itself must
invalidate old cached answers, since they were produced by different logic
and silently replaying them would hide whether the new logic actually
works. AGENT_LOGIC_VERSION must be bumped by hand whenever such a change is
made -- there is no way to detect this automatically without hashing the
entire prompt/tool source, which is more machinery than this needs.

This file's cache (data/agent_runs/layer4_test_cache.jsonl) is a dev/test
cache, deliberately named apart from Layer 8's future <seed>_<run_index>.jsonl
convention so the two don't collide.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.agent.discrepancy import DiscrepancyRecord
from src.agent.resolution import AgentResolution


def record_key(record: DiscrepancyRecord) -> str:
    """Stable identity for a DiscrepancyRecord across runs."""
    if record.order_context is not None:
        return record.order_context.order_id
    return f"unmatched:{record.bank_credits[0].utr}"


def _record_content_hash(record: DiscrepancyRecord) -> str:
    return hashlib.sha256(record.model_dump_json().encode("utf-8")).hexdigest()


def append_run_log(
    log_path: Path, record: DiscrepancyRecord, resolution: AgentResolution, debug_info: dict, logic_version: int
) -> None:
    entry = {
        "record_key": record_key(record),
        "record_content_hash": _record_content_hash(record),
        "logic_version": logic_version,
        "record": json.loads(record.model_dump_json()),
        "resolution": resolution.model_dump(),
        "debug_info": debug_info,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_run_log(log_path: Path) -> dict[str, dict]:
    """Latest entry per record_key -- a cache file may accumulate multiple
    attempts over time (e.g. a record that failed once and was re-run); the
    most recent entry for a given key wins."""
    if not log_path.exists():
        return {}
    latest: dict[str, dict] = {}
    with open(log_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            latest[entry["record_key"]] = entry
    return latest


def total_tokens_used(log_path: Path) -> int:
    """Real cumulative Groq spend across every record's latest cache entry
    -- sums debug_info['tokens_used'] (src/agent/graph.py's per-record
    total, itself summed from Groq's own usage.total_tokens on each live
    call). Entries logged before this field existed contribute 0, not a
    fabricated backfilled estimate -- they were genuinely never measured.
    This is the answer to "how much have we actually spent," replacing the
    ~4,400/~11,700-per-record-run *estimates* used before tokens_used
    existed."""
    return sum(entry["debug_info"].get("tokens_used", 0) for entry in load_run_log(log_path).values())


def count_live_calls_needed(records: list[DiscrepancyRecord], log_path: Path, logic_version: int) -> int:
    """How many of `records` would actually require a live model call --
    i.e. have no entry in log_path whose record_content_hash AND
    logic_version both match -- without calling the model or touching the
    network to find out. Mirrors diagnose_or_replay's own cache-hit check
    exactly, so a batch scored as "0 live calls needed" here really is a
    guaranteed zero live calls when actually run (e.g. the frozen dataset,
    always fully cached, regardless of remaining daily budget).

    Used as the pre-flight input to src.agent.rate_limiter.check_budget_for_batch,
    added after a live debugging session found that hitting the daily quota
    mid-seed-batch crashed with some records already posted to the ledger --
    the fix estimates cost BEFORE any DB write, not just before each
    individual live call (src.agent.graph.diagnose_discrepancy's own
    per-record check_budget, which is too late to avoid partial progress)."""
    cached = load_run_log(log_path)
    needed = 0
    for record in records:
        entry = cached.get(record_key(record))
        if entry is None or entry["record_content_hash"] != _record_content_hash(record) or entry["logic_version"] != logic_version:
            needed += 1
    return needed


def average_real_tokens_per_live_call(log_paths: list[Path], default: float = 7000.0) -> float:
    """Real average tokens_used across every real (nonzero, i.e. actually
    live-called, not cache-replayed) logged invocation across the given log
    files -- never a hardcoded guess. `default` is used only if NO real
    measurement exists anywhere yet (a fresh checkout before any live call
    has ever been logged) -- 7000 is today's real observed average from
    data/agent_runs/frozen_1.jsonl (256,504 tokens / 37 records), used only
    as that one-time bootstrap value, not as an ongoing estimate once real
    data exists."""
    totals = [
        entry["debug_info"]["tokens_used"]
        for path in log_paths
        for entry in load_run_log(path).values()
        if entry["debug_info"].get("tokens_used", 0) > 0
    ]
    if not totals:
        return default
    return sum(totals) / len(totals)


def diagnose_or_replay(
    record: DiscrepancyRecord, log_path: Path, logic_version: int
) -> tuple[AgentResolution, dict, bool]:
    """Returns (resolution, debug_info, was_replayed). Calls the live model
    only if no cache entry exists whose record_content_hash AND
    logic_version both match the current record and current agent logic;
    otherwise reconstructs the result from the cache with zero API calls.
    """
    from src.agent.graph import diagnose_discrepancy  # local import: graph.py doesn't import this module

    cached = load_run_log(log_path).get(record_key(record))
    if (
        cached is not None
        and cached["record_content_hash"] == _record_content_hash(record)
        and cached["logic_version"] == logic_version
    ):
        resolution = AgentResolution.model_validate(cached["resolution"])
        return resolution, cached["debug_info"], True

    resolution, debug_info = diagnose_discrepancy(record)
    append_run_log(log_path, record, resolution, debug_info, logic_version)
    return resolution, debug_info, False
