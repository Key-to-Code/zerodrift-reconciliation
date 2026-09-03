# ZeroDrift

An AI-assisted reconciliation engine that matches orders, gateway settlements, and bank statements ,and is honest about the cases it can't confidently resolve, instead of guessing.

The full scorecard (deterministic results plus the agent's min/median/max across live runs) goes here once Layer 9 wraps and `evaluate.py` has run against the frozen dataset. Until then, this section is a running log of real decisions made while building it ,what changed, why, and what it means for anyone rerunning this.

## Running locally

```
docker compose up -d          # Postgres only -- see docker-compose.yml
source venv/Scripts/activate  # or venv/bin/activate on macOS/Linux
uvicorn src.api.main:app --reload --port 8000
streamlit run src/dashboard/app.py
```

The API applies `src/ledger/db_schema.sql` to the `finance_controller`
database itself on startup if the schema isn't already there (see
`ensure_schema_exists` in `src/ledger/models.py`) -- so a fresh
`docker compose up` (or a `docker compose down -v` reset) doesn't need a
manual `psql -f db_schema.sql` step; the first `uvicorn` start creates it.
`pytest` never touches this database -- it provisions and resets its own
`finance_controller_test` database (`tests/conftest.py`).

## A couple of things worth knowing before you dig in

**The agent was built and tested on an open-source model, not Claude, here's why.** Claude is the model this is actually designed for; the only Razorpay-internal fact this project leans on is that Agent Studio runs on Anthropic's Claude Agent SDK, and that's the track this build targets. But iterating on an agent means a lot of calls ,many records, retries, repeated test runs while things are still being debugged ,and that volume doesn't fit inside free Anthropic Console credits. So development runs on Groq's free tier instead, using `openai/gpt-oss-120b`.

That specific model wasn't the original plan, either. The intent going in was Llama 3.3 70B, but by the time this was built, Groq had quietly dropped it from the models their API actually serves ,it just wasn't in the list anymore when checked. `gpt-oss-120b` was the largest open-weights model still available on the key, so that's what ended up carrying the actual development and testing.

None of this is a permanent architectural choice ,swapping back to Claude for production is a one-line change in `src/agent/graph.py::_build_model()`.

**The agent's evaluation numbers are being gathered across several days, not one sitting ,and that's deliberate, not a shortcut.** A full evaluation run means calling the live agent on every record that needs real judgment (37 of them, per the frozen dataset), and doing that three separate times so the reported result is an honest min/median/max rather than one lucky run dressed up as three. Groq's free tier caps out at 200,000 tokens a day, and three full runs need roughly three times that. So the three run logs -> `data/agent_runs/<seed>_1.jsonl`, `_2.jsonl`, `_3.jsonl` -> genuinely have timestamps days apart. That's the token budget talking, not neglect. The full reasoning is written up in `docs/plan.md`'s Layer 8 section if you want the details.
