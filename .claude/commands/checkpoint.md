Run `pytest tests/ -v` right now and paste the real, complete output — do not summarize
or truncate it.

Then, for the current layer in `docs/plan.md`: go through every acceptance-criteria
checkbox one at a time and state PASS or FAIL, naming the exact test that proves it. If
a checkbox has no test behind it, say so explicitly as FAIL — do not round up.

Finally, scan the code touched so far against `CLAUDE.md`'s integrity rules
specifically: any `float` on a money path, any hardcoded/adjusted test, any number in a
doc or comment that wasn't actually produced by running code, any "distributed" or
"microservices" language, any TDS rate other than 0.1%, any UPI record with nonzero
MDR. Report anything found, even minor.
