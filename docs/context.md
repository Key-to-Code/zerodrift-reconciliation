# Project Context
### Read this before `plan.md`. This file explains the "why" — `plan.md` is the "how."

## What this project is

An AI-assisted financial reconciliation engine, built as a submission for Razorpay's Buildathon, Track 04 ("Agentic Finance / FinOps"). The problem: a merchant's internal order records, a payment gateway's settlement reports, and a bank statement almost never line up perfectly. Timing differences, batched settlements, fee drift, missing tax lines, and genuine orphan transactions all create discrepancies that someone currently has to reconcile by hand. This project automates that — but the actual point being demonstrated isn't "we automated reconciliation," it's **"we built a system that knows the difference between a discrepancy it can safely resolve and one it should honestly flag,"** because that distinction is the entire evaluation thesis of this track.

## Who's grading this and what they actually care about

Track 04's stated bar, paraphrased: verification capacity is the bottleneck, not generation speed. A system that confidently produces a wrong answer is worse than one that says "I don't know, here's why." One cherry-picked correct match proves nothing — judges will rerun the evaluation themselves. The evaluators are engineers who build real financial infrastructure daily; they will ask pointed follow-up questions about any architecture claim, and they will not be impressed by buzzwords they can see through in one question.

This shapes everything in `plan.md`. Every layer exists to produce a **provable, reproducible claim**, not an asserted one.

## The core design decisions, and why they're not up for renegotiation mid-build

- **Money is never a float, anywhere in the system.** Pydantic boundary uses `Decimal`. Polars internals use integer paise. Postgres uses `NUMERIC`. This one discipline, enforced end-to-end, is worth more to the pitch than any single flashy feature — it's the kind of detail that separates someone who's actually thought about financial systems from someone who hasn't.
- **Deterministic fast path first, AI agent only for the genuinely ambiguous tail.** Most discrepancies (clean matches, batched UTRs) don't need an LLM at all — using Polars for those is faster, cheaper, and removes hallucination risk from cases that don't need it. The agent is reserved for cases that actually require judgment (fee drift, missing tax lines, timing anomalies).
- **The agent is bounded and read-only.** Max 3 tool calls, no ledger write access, forced exit to "honest exception" if it can't resolve something confidently. It proposes; a separate gatekeeper decides. This is the single most important safety property in the whole system and the one most likely to be probed hard in Q&A.
- **A real double-entry ledger with a database-level balance trigger**, not just application-level bookkeeping. Unresolved discrepancies still get posted — to a suspense account — so the books stay balanced even when a specific match is still unknown. This mirrors how real accounting systems actually handle uncertainty, rather than just leaving gaps.
- **The dataset includes deliberately adversarial trap records** — pairs of unrelated transactions with coincidentally similar amounts, designed specifically to try to fool the agent into a false match. Catching these correctly is the actual proof of "zero hallucinated matches." Without them, that claim is just an assertion.
- **All benchmark numbers come from `evaluate.py` run against a frozen, labeled dataset (`ground_truth.json`).** Nothing is hand-written into the README. This exists because early drafts of this plan (from a different AI tool, before this rebuild) included a pre-written "results" scorecard before any code existed — that was flagged as a serious credibility risk and is explicitly not being repeated here.

## Architecture stance — say this the same way every time

This is a **modular monolith**, not a distributed system or microservices architecture. One Python process, one Postgres instance, in-process function calls between modules, single ACID transactions for ledger writes. This was a deliberate choice, not a limitation: a message broker, service mesh, and distributed transaction handling (Saga pattern, compensating transactions, idempotency across network boundaries) would add real operational risk for a 100-record batch with zero throughput benefit. The module boundaries (fast-match is stateless and CPU-bound, the agent is I/O-bound, the ledger owns transactional invariants) are kept clean specifically so a future split into real services is plausible — but nothing here is distributed today, and the code, comments, and pitch materials should never imply otherwise.

If the agent or any generated content refers to this system as "distributed," "microservices," or claims cross-service failure handling that doesn't exist, that's a mistake — flag it and correct it.

## Framework choice — LangGraph, not Claude Agent SDK, and why that's stated explicitly

Razorpay's own production agent platform (Agent Studio, launched March 2026) runs on Anthropic's Claude Agent SDK — this is a verified, citable fact. It might seem like the "obvious" choice to mirror that. It's deliberately not used here: Agent SDK is built around Claude Code's open-ended autonomous-work model (Read/Write/Edit/Bash-style tools, permission hooks), while this project needs a strict, explicit, finite-state diagnostic loop with a hardcoded 3-tool-call ceiling and a deterministic exit condition. LangGraph's explicit graph nodes are a better structural fit for that guarantee. This tradeoff — chosen deliberately, not out of unfamiliarity with Agent SDK — should be stated plainly in the README as evidence of research and judgment, not hidden.

## What NOT to add, even if it seems like it would look more impressive

Next.js (Streamlit is the one and only frontend), pgvector (rapidfuzz string matching is sufficient at this scale), full OpenTelemetry tracing (structured JSON logging via Structlog covers the audit-trail need at a fraction of the setup cost), Kafka/Spark/any streaming infrastructure, and any unverified claims about Razorpay's internal tech stack. Every one of these was proposed at some point during planning and deliberately cut — re-adding them under time pressure or because they "sound better" is the exact failure mode this project is trying to avoid. If in doubt, prefer the version of a decision that can be fully explained and defended over the version that sounds more advanced.

## The single most important artifact in the whole build

`ground_truth.json`, produced in Layer 1. Every claim the finished product makes — match rate, zero false positives, zero hallucinated matches, honest exception handling — is only as credible as this file. If time runs short, protect Layer 1 and Layer 2 (dataset + fast path) before anything else; a smaller but fully verifiable system beats a larger system with unverifiable claims, every time, for this specific evaluation.

## Explicit instruction to the coding agent: do not fake results, ever

This needs to be stated directly, because it's the single easiest way to fail this project without realizing it. If at any point a test is failing, a metric isn't hitting its target, or a layer isn't working correctly:

- **Do not hardcode a passing test to make it green.** A test exists to verify behavior — editing the test instead of fixing the code is the same thing as lying about the result.
- **Do not write a plausible-looking number into `evaluate.py`'s output, the README, or anywhere else unless that exact number was actually produced by running the actual code against the actual `ground_truth.json`.** If the real match rate is 54% instead of the hoped-for 65%, report 54% and say why — that is a more credible submission than a fabricated 100%, not a less credible one.
- **Do not fabricate, mock, or pre-bake a "successful" agent response** to make the diagnostic loop look like it resolved something it didn't actually resolve. If the agent fails on a case, that failure is real information — surface it, don't paper over it.
- **Do not invent sample data that isn't clearly and explicitly labeled as synthetic**, and never present output from an untested code path as if it were verified.
- **If something isn't working, say so plainly and stop to report it**, rather than producing an artifact that looks finished but silently isn't. A half-built layer that's honestly flagged as incomplete is worth more than a fully-built-looking layer that's quietly faking its output.

The entire point of this project, and the entire evaluation thesis of the track it's being built for, is that a system which honestly says "I don't know" is more valuable than one that confidently produces a wrong or fabricated answer. That standard applies to the build process itself, not just the finished product's behavior. If the agent building this project fakes a result to hit a target, it has broken the exact thing the project is trying to prove.

## Addendum (plan v2): reproducibility claim, stated precisely

An earlier draft of `plan.md` implied `evaluate.py` would reproduce one exact scorecard,
including the agent's numbers. That's not true and saying it is would be the single
biggest self-inflicted credibility risk in the whole submission — the agent path is an
LLM and is not deterministic, even at temperature 0. `plan.md` v2 fixes this: the
deterministic path (generator, fast path, ledger, paise round-trip) is byte-reproducible
and reported as an exact number; the agent path is run 3 times and reported as a range
with variance. Every agent run is cached to disk so a `--replay` mode can re-score
without calling the model again — this is also the offline fallback if the demo venue's
network or the API has issues. State this distinction plainly in the README rather than
collapsing it into one falsely-precise number.

## End goal beyond the buildathon

This project is also intended to support a resume and internship applications (Razorpay's own internship track, and separately, other financial-sector internships such as TD Bank). That means the code and docs should stay defensible under follow-up questioning, not just impressive at first glance — assume every design decision may be asked about directly in an interview, and prefer decisions that can be explained in two honest sentences over ones that only sound good in a pitch.
