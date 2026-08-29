"""Deterministic synthetic dataset generator.

CLI: python -m src.data.generator --records 100 --seed 42 --out data/batch_100

Same seed must produce byte-identical output on every run, so nothing here
may read wall-clock time or iterate a plain set/dict-with-random-keys in a
way that affects emitted order -- all randomness comes from a single seeded
random.Random instance, and dates are anchored to a fixed window hardcoded
below (docs/plan.md Layer 1 Addendum A5), never date.today().
"""
import argparse
import json
import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path
from typing import Optional

from src.common.calendar import add_business_days
from src.common.money import to_paise, from_paise
from src.data.models import BankStatementLine, GatewaySettlement, GroundTruthEntry, InternalOrder

# ---------------------------------------------------------------------------
# Domain constants -- verified, see CLAUDE.md Sec.4. Do not re-derive/guess.
# ---------------------------------------------------------------------------

GST_RATE = Decimal("0.18")
TDS_RATE = Decimal("0.001")  # Section 194-O, reduced from 1% effective 01-Oct-2024.

# Synthetic, illustrative MDR schedule used only to generate plausible fee
# amounts -- not a verified real-world rate card. UPI is nil by regulation
# (CLAUDE.md Sec.4); the other rates are made up for realism only.
STANDARD_MDR_RATES = {
    "upi": Decimal("0.0000"),
    "debit_card": Decimal("0.0040"),
    "credit_card": Decimal("0.0150"),
    "netbanking": Decimal("0.0100"),
    "amex": Decimal("0.0250"),
}

RAILS = ["upi", "credit_card", "debit_card", "netbanking", "amex"]
RAIL_WEIGHTS = [40, 25, 15, 10, 10]

# Fixed dataset window (docs/plan.md Layer 1 Addendum A5) -- never derived
# from date.today(), so determinism holds regardless of when this is run.
BATCH_START = date(2025, 1, 6)
BATCH_END = date(2025, 2, 2)

CATEGORY_COUNTS = {
    "clean_match": 53,
    "utr_batch": 10,
    "cutoff_drift": 5,
    "fee_drift": 7,
    "missing_tax_line": 5,
    "orphan": 8,
    "adversarial_trap": 5,
    "short_settlement": 2,
    "duplicate_credit": 2,
    "refund_clawback": 3,
}

UTR_BATCH_PARTITIONS = [[2, 4, 4], [3, 3, 4], [2, 3, 2, 3], [4, 3, 3], [2, 2, 3, 3]]

NARRATION_TEMPLATES = [
    lambda utr: f"NEFT-{utr}-SETTLE",
    lambda utr: f"IMPS/RZPY/{utr[-6:]}/xx",
    lambda utr: f"RZPY SETTLEMENT UTR {utr} REF{utr[-4:]}",
    lambda utr: f"UPI/RZPY/{utr}/SETTLE",
    lambda utr: f"NEFT RZPYSETL {utr}",
]

CENT = Decimal("0.01")


@dataclass
class GeneratedBatch:
    orders: list[InternalOrder]
    settlements: list[GatewaySettlement]
    bank_lines: list[BankStatementLine]
    ground_truth: list[GroundTruthEntry]


def _round(d: Decimal) -> Decimal:
    return d.quantize(CENT, rounding=ROUND_HALF_UP)


def _scaled_counts(num_records: int) -> dict[str, int]:
    if num_records == 100:
        return dict(CATEGORY_COUNTS)
    ratio = num_records / 100
    scaled = {k: max(0, round(v * ratio)) for k, v in CATEGORY_COUNTS.items()}
    if scaled["adversarial_trap"] == 0:
        scaled["adversarial_trap"] = 1
    diff = num_records - sum(scaled.values())
    scaled["clean_match"] += diff
    if scaled["clean_match"] < 0:
        raise ValueError(f"num_records={num_records} too small to satisfy category minimums")
    return scaled


def _new_utr(rng: random.Random, used_utrs: set[str]) -> str:
    while True:
        candidate = f"UTR{rng.randint(1_000_000_000, 9_999_999_999)}"
        if candidate not in used_utrs:
            used_utrs.add(candidate)
            return candidate


def _make_narration(rng: random.Random, utr: str) -> str:
    template = rng.choice(NARRATION_TEMPLATES)
    return template(utr)


def _weighted_rail(rng: random.Random) -> str:
    return rng.choices(RAILS, weights=RAIL_WEIGHTS, k=1)[0]


def _random_date_in_window(rng: random.Random) -> date:
    span = (BATCH_END - BATCH_START).days
    offset = rng.randint(0, span)
    return BATCH_START + timedelta(days=offset)


def _random_datetime_in_window(rng: random.Random) -> datetime:
    d = _random_date_in_window(rng)
    return datetime(d.year, d.month, d.day, rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59))


def _cutoff_datetime_in_window(rng: random.Random) -> datetime:
    d = _random_date_in_window(rng)
    return datetime(d.year, d.month, d.day, 23, rng.randint(45, 59), rng.randint(0, 59))


def _settlement_window_business_days(rail: str, is_international: bool) -> int:
    if is_international:
        return 3
    if rail == "upi":
        return 1
    return 2


def _compute_settlement_fields(gross: Decimal, rail: str) -> tuple[Decimal, Decimal, Decimal, Decimal]:
    standard_rate = STANDARD_MDR_RATES[rail]
    mdr = _round(gross * standard_rate) if rail != "upi" else Decimal("0.00")
    gst = _round(mdr * GST_RATE) if mdr > 0 else Decimal("0.00")
    tds = _round(gross * TDS_RATE)
    net = gross - mdr - gst - tds
    return mdr, gst, tds, net


def _replace(model, **updates):
    data = model.model_dump()
    data.update(updates)
    return type(model)(**data)


class _Counter:
    def __init__(self):
        self.n = 0

    def next(self) -> int:
        self.n += 1
        return self.n


def _make_clean_flow(
    rng: random.Random,
    used_utrs: set[str],
    counter: _Counter,
    rail: str,
    order_dt: datetime,
    is_international: bool = False,
    gross: Optional[Decimal] = None,
) -> tuple[InternalOrder, GatewaySettlement, BankStatementLine]:
    num = counter.next()
    order_id = f"ORD{1000 + num}"
    customer_id = f"CUST{1000 + num}"
    if gross is None:
        gross = from_paise(rng.randint(50_000, 500_000))
    order = InternalOrder(
        order_id=order_id,
        gross_amount=gross,
        customer_id=customer_id,
        payment_method=rail,
        timestamp=order_dt,
    )
    mdr, gst, tds, net = _compute_settlement_fields(gross, rail)
    utr = _new_utr(rng, used_utrs)
    window = _settlement_window_business_days(rail, is_international)
    settlement_date = add_business_days(order_dt.date(), window)
    settlement = GatewaySettlement(
        payment_id=f"PAY{1000 + num}",
        order_id=order_id,
        gross_amount=gross,
        payment_method=rail,
        mdr=mdr,
        gst_on_mdr=gst,
        tds_194o=tds,
        net_amount=net,
        utr=utr,
        settlement_date=settlement_date,
        is_international=is_international,
    )
    bank_line = BankStatementLine(
        utr=utr,
        credited_amount=net,
        value_date=settlement_date,
        narration=_make_narration(rng, utr),
    )
    return order, settlement, bank_line


def generate_batch(num_records: int, seed: int) -> GeneratedBatch:
    rng = random.Random(seed)
    counts = _scaled_counts(num_records)
    used_utrs: set[str] = set()
    counter = _Counter()

    orders: list[InternalOrder] = []
    settlements: list[GatewaySettlement] = []
    bank_lines: list[BankStatementLine] = []
    ground_truth: list[GroundTruthEntry] = []

    # -- clean_match, generated first so adversarial_trap can pick a "twin" --
    clean_match_orders: list[InternalOrder] = []
    for _ in range(counts["clean_match"]):
        rail = _weighted_rail(rng)
        order_dt = _random_datetime_in_window(rng)
        order, settlement, bank_line = _make_clean_flow(rng, used_utrs, counter, rail, order_dt)
        orders.append(order)
        settlements.append(settlement)
        bank_lines.append(bank_line)
        clean_match_orders.append(order)
        ground_truth.append(
            GroundTruthEntry(
                order_id=order.order_id,
                category="clean_match",
                expected_resolution="fast_path",
                notes="Standard settlement; MDR/GST/TDS all consistent with the rate table.",
            )
        )

    # -- utr_batch: grouped orders sharing one UTR, one lump-sum bank credit --
    for size in rng.choice(UTR_BATCH_PARTITIONS):
        rail = _weighted_rail(rng)
        base_date = _random_date_in_window(rng)
        batch_utr = _new_utr(rng, used_utrs)
        window = _settlement_window_business_days(rail, False)
        settlement_date = add_business_days(base_date, window)
        group_settlements: list[GatewaySettlement] = []
        for _ in range(size):
            num = counter.next()
            order_dt = datetime(
                base_date.year, base_date.month, base_date.day,
                rng.randint(0, 23), rng.randint(0, 59), rng.randint(0, 59),
            )
            gross = from_paise(rng.randint(50_000, 500_000))
            order = InternalOrder(
                order_id=f"ORD{1000 + num}",
                gross_amount=gross,
                customer_id=f"CUST{1000 + num}",
                payment_method=rail,
                timestamp=order_dt,
            )
            mdr, gst, tds, net = _compute_settlement_fields(gross, rail)
            settlement = GatewaySettlement(
                payment_id=f"PAY{1000 + num}",
                order_id=order.order_id,
                gross_amount=gross,
                payment_method=rail,
                mdr=mdr,
                gst_on_mdr=gst,
                tds_194o=tds,
                net_amount=net,
                utr=batch_utr,
                settlement_date=settlement_date,
                is_international=False,
            )
            orders.append(order)
            settlements.append(settlement)
            group_settlements.append(settlement)
            ground_truth.append(
                GroundTruthEntry(
                    order_id=order.order_id,
                    category="utr_batch",
                    expected_resolution="fast_path",
                    notes=(
                        f"Grouped under UTR {batch_utr} with {size - 1} sibling order(s); "
                        "net amounts split via largest-remainder allocation (Layer 3)."
                    ),
                )
            )
        total_net = sum((s.net_amount for s in group_settlements), Decimal("0.00"))
        bank_lines.append(
            BankStatementLine(
                utr=batch_utr,
                credited_amount=total_net,
                value_date=settlement_date,
                narration=_make_narration(rng, batch_utr),
            )
        )

    # -- cutoff_drift: near-midnight order, settlement pushed +1 business day --
    for _ in range(counts["cutoff_drift"]):
        rail = _weighted_rail(rng)
        order_dt = _cutoff_datetime_in_window(rng)
        order, settlement, bank_line = _make_clean_flow(rng, used_utrs, counter, rail, order_dt)
        base_window = _settlement_window_business_days(rail, False)
        shifted_date = add_business_days(order_dt.date(), base_window + 1)
        settlement = _replace(settlement, settlement_date=shifted_date)
        bank_line = _replace(bank_line, value_date=shifted_date)
        root_cause = "CUTOFF_T1" if base_window == 1 else "CUTOFF_T2"
        orders.append(order)
        settlements.append(settlement)
        bank_lines.append(bank_line)
        ground_truth.append(
            GroundTruthEntry(
                order_id=order.order_id,
                category="cutoff_drift",
                expected_resolution="agent_resolved",
                expected_root_cause_code=root_cause,
                expected_delta_paise=0,
                notes=(
                    f"Order placed at {order_dt.time()} IST; cutoff pushed settlement to "
                    f"{shifted_date} (+1 business day beyond standard T+{base_window})."
                ),
            )
        )

    # -- fee_drift: amex/international surcharge on top of the standard rate --
    for _ in range(counts["fee_drift"]):
        is_intl = rng.random() < 0.4
        rail = rng.choice(["amex", "credit_card"]) if is_intl else "amex"
        order_dt = _random_datetime_in_window(rng)
        order, settlement, bank_line = _make_clean_flow(
            rng, used_utrs, counter, rail, order_dt, is_international=is_intl
        )
        surcharge_bps = rng.randint(200, 300)  # 2.00%-3.00%, randomized per spec
        surcharge_rate = Decimal(surcharge_bps) / Decimal(10_000)
        extra_mdr = _round(settlement.gross_amount * surcharge_rate)
        new_mdr = settlement.mdr + extra_mdr
        new_gst = _round(new_mdr * GST_RATE)
        new_net = settlement.gross_amount - new_mdr - new_gst - settlement.tds_194o
        settlement = _replace(settlement, mdr=new_mdr, gst_on_mdr=new_gst, net_amount=new_net)
        bank_line = _replace(bank_line, credited_amount=new_net)
        root_cause = "INTL_MARKUP" if is_intl else "AMEX_SURCHARGE"
        orders.append(order)
        settlements.append(settlement)
        bank_lines.append(bank_line)
        ground_truth.append(
            GroundTruthEntry(
                order_id=order.order_id,
                category="fee_drift",
                expected_resolution="agent_resolved",
                expected_root_cause_code=root_cause,
                expected_delta_paise=to_paise(extra_mdr),
                notes=f"{rail} surcharge of {surcharge_rate * 100}% not in standard MDR table (+Rs {extra_mdr}).",
            )
        )

    # -- missing_tax_line: GST or TDS line zeroed in the settlement feed --
    # Never on UPI: GST-on-MDR is already legitimately 0 there (nothing to
    # omit), so a "missing" line would be an impossible transaction (CLAUDE.md Sec.4).
    non_upi_rails = [r for r in RAILS if r != "upi"]
    for _ in range(counts["missing_tax_line"]):
        rail = rng.choice(non_upi_rails)
        order_dt = _random_datetime_in_window(rng)
        order, settlement, bank_line = _make_clean_flow(rng, used_utrs, counter, rail, order_dt)
        omit_gst = rng.random() < 0.5
        if omit_gst:
            omitted = settlement.gst_on_mdr
            settlement = _replace(settlement, gst_on_mdr=Decimal("0.00"), net_amount=settlement.net_amount + omitted)
            root_cause = "MISSING_GST"
        else:
            omitted = settlement.tds_194o
            settlement = _replace(settlement, tds_194o=Decimal("0.00"), net_amount=settlement.net_amount + omitted)
            root_cause = "MISSING_TDS"
        bank_line = _replace(bank_line, credited_amount=settlement.net_amount)
        orders.append(order)
        settlements.append(settlement)
        bank_lines.append(bank_line)
        ground_truth.append(
            GroundTruthEntry(
                order_id=order.order_id,
                category="missing_tax_line",
                expected_resolution="agent_resolved",
                expected_root_cause_code=root_cause,
                expected_delta_paise=to_paise(omitted),
                notes=f"{root_cause.replace('MISSING_', '')} line omitted from the settlement feed.",
            )
        )

    # -- short_settlement: settlement exists, no bank credit ever appears --
    for _ in range(counts["short_settlement"]):
        rail = _weighted_rail(rng)
        order_dt = _random_datetime_in_window(rng)
        order, settlement, _unused_bank_line = _make_clean_flow(rng, used_utrs, counter, rail, order_dt)
        orders.append(order)
        settlements.append(settlement)
        ground_truth.append(
            GroundTruthEntry(
                order_id=order.order_id,
                category="short_settlement",
                expected_resolution="honest_exception",
                notes="Settlement recorded with a valid UTR, but no bank credit for it ever appears.",
            )
        )

    # -- duplicate_credit: same UTR credited twice in the bank statement --
    for _ in range(counts["duplicate_credit"]):
        rail = _weighted_rail(rng)
        order_dt = _random_datetime_in_window(rng)
        order, settlement, bank_line = _make_clean_flow(rng, used_utrs, counter, rail, order_dt)
        orders.append(order)
        settlements.append(settlement)
        bank_lines.append(bank_line)
        duplicate_line = _replace(bank_line, narration=_make_narration(rng, settlement.utr))
        bank_lines.append(duplicate_line)
        ground_truth.append(
            GroundTruthEntry(
                order_id=order.order_id,
                category="duplicate_credit",
                expected_resolution="honest_exception",
                notes=f"UTR {settlement.utr} was credited twice in the bank statement.",
            )
        )

    # -- refund_clawback: gross debited/credited normally, MDR never reversed --
    for _ in range(counts["refund_clawback"]):
        rail = _weighted_rail(rng)
        order_dt = _random_datetime_in_window(rng)
        order, settlement, bank_line = _make_clean_flow(rng, used_utrs, counter, rail, order_dt)
        refund_bps = rng.randint(1000, 6000)  # 10%-60% of gross refunded
        refund_amount = _round(order.gross_amount * Decimal(refund_bps) / Decimal(10_000))
        if refund_amount <= 0:
            refund_amount = Decimal("1.00")
        if refund_amount >= order.gross_amount:
            refund_amount = order.gross_amount - Decimal("1.00")
        order = _replace(order, refund_amount=refund_amount)
        orders.append(order)
        settlements.append(settlement)
        bank_lines.append(bank_line)
        ground_truth.append(
            GroundTruthEntry(
                order_id=order.order_id,
                category="refund_clawback",
                expected_resolution="agent_resolved",
                expected_root_cause_code="REFUND_NO_MDR_REVERSAL",
                expected_delta_paise=to_paise(refund_amount),
                notes=(
                    f"Customer refund of Rs {refund_amount} debits revenue but does not "
                    "reverse the MDR already deducted on the original gross amount."
                ),
            )
        )

    # -- orphan: bank credit with no settlement/UTR anywhere in the batch --
    for _ in range(counts["orphan"]):
        utr = _new_utr(rng, used_utrs)
        credited = from_paise(rng.randint(50_000, 500_000))
        value_date = _random_date_in_window(rng)
        bank_lines.append(
            BankStatementLine(
                utr=utr,
                credited_amount=credited,
                value_date=value_date,
                narration=_make_narration(rng, utr),
            )
        )
        ground_truth.append(
            GroundTruthEntry(
                order_id=f"UNMATCHED_BANK_{utr}",
                category="orphan",
                expected_resolution="honest_exception",
                notes="Bank credit with no matching settlement anywhere in the batch.",
            )
        )

    # -- adversarial_trap: unrelated bank credit coincidentally similar to a
    # real clean_match order's amount/date -- must not be force-matched.
    for _ in range(counts["adversarial_trap"]):
        twin_order = rng.choice(clean_match_orders)
        twin_settlement = next(s for s in settlements if s.order_id == twin_order.order_id)
        jitter_paise = rng.randint(-200, 200)
        decoy_paise = to_paise(twin_settlement.net_amount) + jitter_paise
        if decoy_paise <= 0:
            decoy_paise = to_paise(twin_settlement.net_amount)
        utr = _new_utr(rng, used_utrs)
        value_date = twin_settlement.settlement_date + timedelta(days=rng.choice([-1, 0, 1]))
        bank_lines.append(
            BankStatementLine(
                utr=utr,
                credited_amount=from_paise(decoy_paise),
                value_date=value_date,
                narration=_make_narration(rng, utr),
            )
        )
        ground_truth.append(
            GroundTruthEntry(
                order_id=f"UNMATCHED_BANK_{utr}",
                category="adversarial_trap",
                expected_resolution="honest_exception",
                notes=(
                    f"Amount/date coincidentally close to {twin_order.order_id}, but shares "
                    "no real UTR or order linkage -- must not be force-matched."
                ),
            )
        )

    return GeneratedBatch(orders=orders, settlements=settlements, bank_lines=bank_lines, ground_truth=ground_truth)


def _json_default(obj):
    if isinstance(obj, Decimal):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")


def write_batch(batch: GeneratedBatch, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "internal_orders.json", [o.model_dump(mode="python") for o in batch.orders])
    _write_json(out_dir / "gateway_settlement.json", [s.model_dump(mode="python") for s in batch.settlements])
    _write_json(out_dir / "bank_statement.json", [b.model_dump(mode="python") for b in batch.bank_lines])
    _write_json(out_dir / "ground_truth.json", [g.model_dump(mode="python") for g in batch.ground_truth])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic reconciliation dataset.")
    parser.add_argument("--records", type=int, default=100)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    batch = generate_batch(num_records=args.records, seed=args.seed)
    write_batch(batch, Path(args.out))
    print(f"Wrote {args.records}-record batch (seed={args.seed}) to {args.out}")


if __name__ == "__main__":
    main()
