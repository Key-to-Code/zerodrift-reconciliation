You did not write the code you are about to review — treat it that way. Read
`CLAUDE.md`, `docs/plan.md`, and the implementation for Layer $ARGUMENTS with a
skeptical, adversarial eye. You are specifically looking for:

1. Tests that assert against a hardcoded expected value rather than actual behaviour
   derived from the acceptance criteria (a test that would still pass if the
   implementation were silently broken).
2. Any `float` anywhere on a code path that touches a monetary amount.
3. Any acceptance criterion marked complete in a prior commit message or report that
   has no corresponding test, or where the test doesn't actually exercise the claimed
   behavior.
4. Domain-fact drift: TDS rate other than 0.1%, any UPI record with nonzero MDR/GST,
   GST or TDS posted to a liability account instead of an asset, a date-window check
   using raw calendar days instead of `src/common/calendar.py`.
5. Language anywhere (code comments, docstrings, README) describing the system as
   distributed, microservices, or claiming cross-service failure handling.
6. Any UTR batch allocation that doesn't sum exactly to the bank credit, or that
   silently absorbs a rounding residual into one order's amount instead of posting it
   to `ROUNDING_DIFFERENCE`.
7. Any place a number appears in a doc, README, or log message that you cannot trace
   back to an actual `pytest` or `evaluate.py` run.

For each issue found, quote the exact file and line, explain why it's a problem, and
propose the fix. If you find nothing, say so plainly rather than padding the review —
but re-check item 1 and item 7 twice before concluding that.
