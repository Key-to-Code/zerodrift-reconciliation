"""Layer 3: largest-remainder allocation for splitting a UTR batch's
lump-sum bank credit across N orders without leaving unallocated paise
(CLAUDE.md Sec.6).

`shares` are each order's own known, factual net entitlement in integer
paise -- e.g. `GatewaySettlement.net_amount_paise` -- which in the ordinary
case already sums exactly to the bank's lump-sum credit. `allocate_utr_batch`
treats them as proportional weights and rescales them to sum exactly to
`total_paise` via floor-then-largest-remainder, so it is a no-op (returns the
shares unchanged) whenever they already sum to `total_paise`, and only
redistributes paise when a small, genuine gap exists between what the
settlements expected and what the bank actually credited.
"""
from __future__ import annotations


class AllocationResidualTooLarge(Exception):
    """Raised by `assert_allocation_gap_within_cap` when the gap between
    `total_paise` and the sum of `shares` exceeds the +/- len(shares) paise
    cap. A gap this size is not ordinary per-order rounding -- it signals an
    upstream allocation bug -- and must never be silently swept into
    ROUNDING_DIFFERENCE."""


def assert_allocation_gap_within_cap(total_paise: int, shares: list[tuple[str, int]]) -> None:
    """Guard for the specific ledger use of `allocate_utr_batch`, where
    `shares` are each order's own known, factual net amount and should
    already approximately equal `total_paise` (the real UTR bank credit).
    Raises if that gap exceeds the +/- len(shares) paise cap.

    This is deliberately NOT enforced inside `allocate_utr_batch` itself --
    that function is a general largest-remainder apportionment algorithm
    also used to split a total by arbitrary proportional weights unrelated
    in scale to the total (e.g. splitting a flat batch fee equally three
    ways is `shares=[("A",1),("B",1),("C",1)]` against a much larger
    `total_paise`), which this cap would incorrectly reject.
    """
    share_total = sum(share for _, share in shares)
    gap = total_paise - share_total
    if abs(gap) > len(shares):
        raise AllocationResidualTooLarge(
            f"gap of {gap} paise between total_paise ({total_paise}) and sum(shares) "
            f"({share_total}) exceeds the +/-{len(shares)} cap for {len(shares)} shares"
        )


def allocate_utr_batch(total_paise: int, shares: list[tuple[str, int]]) -> dict[str, int]:
    """
    shares: list of (order_id, exact_net_share_paise) BEFORE rounding -- each
    order's proportional entitlement, rescaled below to sum exactly to
    `total_paise`.

    Returns {order_id: allocated_paise} summing exactly to total_paise.
    """
    if not shares:
        raise ValueError("shares must be non-empty")

    share_total = sum(share for _, share in shares)
    if share_total <= 0:
        raise ValueError("shares must sum to a positive amount")

    floors: dict[str, int] = {}
    remainder_numerators: dict[str, int] = {}
    for order_id, share in shares:
        scaled_numerator = share * total_paise
        floor_paise, remainder_numerator = divmod(scaled_numerator, share_total)
        floors[order_id] = floor_paise
        remainder_numerators[order_id] = remainder_numerator

    allocated = dict(floors)
    leftover = total_paise - sum(floors.values())
    assert 0 <= leftover < len(shares), (
        f"internal invariant violated: leftover {leftover} out of bounds for {len(shares)} shares"
    )

    order_ids_in_input_order = [order_id for order_id, _ in shares]
    order_ids_by_remainder_desc = sorted(
        order_ids_in_input_order, key=lambda oid: remainder_numerators[oid], reverse=True
    )
    for order_id in order_ids_by_remainder_desc[:leftover]:
        allocated[order_id] += 1

    assert sum(allocated.values()) == total_paise
    return allocated
