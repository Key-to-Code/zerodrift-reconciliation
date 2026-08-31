"""Dev iteration tool: diagnose ONE record via diagnose_or_replay, without
touching the rest of its category.

Built after repeated AGENT_LOGIC_VERSION bumps (each invalidating the whole
Layer 4 test cache by design -- see src/agent/run_log.py's module docstring,
that coarseness is deliberate and stays) forced a full 17-record live
re-run to check whether a single-record prompt fix worked, which burned
through two Groq API keys' daily quota in one session (observed cost:
~11,700 tokens/record-run on the fee_drift/missing_tax_line/cutoff_drift
categories, not the ~4,400 tokens/record-run originally estimated).

Workflow this enables: after a prompt/tool fix, run this against ONLY the
record that was wrong. Only if that passes is a full category/suite re-run
(which re-verifies every record, catching any regression the fix might have
caused elsewhere) worth spending quota on.

Reuses build_settlement_discrepancy_queue / build_unmatched_bank_line_queue /
diagnose_or_replay directly -- no diagnostic logic is reimplemented here.

Usage:
    python scripts/diagnose_one.py ORD1069          # a settled discrepancy, by order_id
    python scripts/diagnose_one.py UTR1234567890    # an unmatched bank line, by UTR
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from src.agent.discrepancy import build_settlement_discrepancy_queue, build_unmatched_bank_line_queue
from src.agent.graph import AGENT_LOGIC_VERSION
from src.agent.run_log import diagnose_or_replay
from src.data.models import BankStatementLine, GatewaySettlement, GroundTruthEntry, InternalOrder
from src.matching.schema import bank_lines_to_frame, orders_to_frame, settlements_to_frame

FROZEN_DIR = _ROOT / "data" / "challenge_batch_100"
CACHE_PATH = _ROOT / "data" / "agent_runs" / "layer4_test_cache.jsonl"


def load_frozen_frames():
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
    ground_truth = {
        e.order_id: e
        for e in (
            GroundTruthEntry.model_validate(d)
            for d in json.loads((FROZEN_DIR / "ground_truth.json").read_text(encoding="utf-8"))
        )
    }
    return orders_to_frame(orders), settlements_to_frame(settlements), bank_lines_to_frame(bank_lines), ground_truth


def find_record(identifier: str, orders_df, settlements_df, bank_df):
    """Returns (record, ground_truth_key) or (None, None) if not found in
    either queue. Checked as a settled order_id first, then as an unmatched
    bank line's UTR."""
    for record in build_settlement_discrepancy_queue(orders_df, settlements_df, bank_df):
        if record.order_context and record.order_context.order_id == identifier:
            return record, record.order_context.order_id

    for record in build_unmatched_bank_line_queue(orders_df, settlements_df, bank_df):
        if record.bank_credits and record.bank_credits[0].utr == identifier:
            return record, f"UNMATCHED_BANK_{record.bank_credits[0].utr}"

    return None, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("identifier", help="order_id (settled) or UTR (unmatched bank line)")
    args = parser.parse_args()

    orders_df, settlements_df, bank_df, ground_truth = load_frozen_frames()
    record, gt_key = find_record(args.identifier, orders_df, settlements_df, bank_df)

    if record is None:
        print(f"No discrepancy record found for identifier {args.identifier!r} in either queue.")
        sys.exit(1)

    resolution, debug, replayed = diagnose_or_replay(record, CACHE_PATH, AGENT_LOGIC_VERSION)

    print(f"identifier: {args.identifier}")
    print(f"replayed_from_cache: {replayed}")
    print(f"resolution: {resolution.model_dump_json(indent=2)}")
    print(f"debug_info: {json.dumps(debug, indent=2)}")

    gt = ground_truth.get(gt_key)
    if gt is None:
        print(f"ground_truth: NOT FOUND for key {gt_key!r}")
        return

    print(f"ground_truth: root_cause_code={gt.expected_root_cause_code!r} delta_paise={gt.expected_delta_paise!r}")
    code_ok = resolution.root_cause_code == gt.expected_root_cause_code
    delta_ok = gt.expected_delta_paise is None or resolution.quantified_delta_paise == gt.expected_delta_paise
    print(f"MATCH: root_cause_code={code_ok} delta={delta_ok}")


if __name__ == "__main__":
    main()
