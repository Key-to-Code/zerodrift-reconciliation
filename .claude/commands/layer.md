Read `CLAUDE.md` in full, and Layer $ARGUMENTS of `docs/plan.md` in full.

Do the following, in order, and stop after step 2 to wait for my approval:

1. Restate Layer $ARGUMENTS's acceptance criteria as a numbered checklist, in your own
   words, to confirm you've actually understood them rather than just copying them.
2. List every test you intend to write in `tests/` for this layer, mapped 1:1 to an
   acceptance criterion (note in parentheses which criterion each test proves). If a
   criterion can't be turned into a concrete test, say so explicitly and propose how
   you'll verify it instead. Flag any part of the spec in `docs/plan.md` that is
   ambiguous or underspecified as a question for me, rather than guessing silently.

Wait for my go-ahead before writing any implementation code. Once approved:

3. Write the tests first.
4. Implement until they pass honestly — do not edit a test to make it pass.
5. Run `pytest tests/ -v` — the **whole** suite, not just this layer's new file — and
   paste the real output.
6. Report acceptance-criteria status as PASS/FAIL, each with the specific test that
   proves it. Do not mark anything PASS that no test actually covers.
7. `git commit` with the layer number in the message.
