# AI Finance Controller

Full scorecard (deterministic block + agent min/median/max block, per CLAUDE.md Sec.5)
is written in Layer 9, once every layer is built and evaluate.py has produced real
numbers against the frozen dataset. This file currently holds only development notes
that are true as of the layer that wrote them.

## Development notes

**Layer 4 (agent) model used during development is not Claude.** Claude is the
intended production model -- the only permitted Razorpay-internal claim in this
project is that Agent Studio runs on Anthropic's Claude Agent SDK, and this build
targets that track. Development and testing instead run against Groq-hosted
`openai/gpt-oss-120b` (via `langchain-groq`), for two reasons:

1. **Cost.** Anthropic Console credits aren't available for the iteration volume
   Layer 4 needs (many records x multiple retries x repeated test runs while
   building); Groq's free tier covers that volume.
2. **Model availability within Groq.** Llama 3.3 70B, the originally intended
   dev model, was checked against the live Groq API key's `/openai/v1/models`
   listing at build time and was not present -- Groq appears to have
   deprecated/rotated it out of the free-tier offering since it was documented.
   `openai/gpt-oss-120b` is the largest open-weights chat model this key actually
   has access to, so it was used instead.

Swapping the production model to Claude is a one-line change in
`src/agent/graph.py::_build_model()`.

**Layer 8's 3-run agent evaluation is gathered across a multi-day window, not one
sitting.** Groq's free tier caps at 200,000 tokens/day; a full evaluation run
invokes the live agent on all 37 records with `expected_resolution` in
`{agent_resolved, honest_exception}` (per `ground_truth.json`), and reporting a
genuine min/median/max across 3 *independent* live runs (CLAUDE.md Sec.5 — not one
run cached and replayed twice) needs roughly 3x that per day's budget on a single
free key. So `data/agent_runs/<seed>_1.jsonl`, `_2.jsonl`, `_3.jsonl` carry
timestamps days apart by design, not by accident — see `docs/plan.md`'s Layer 8
section for the full methodology note.
