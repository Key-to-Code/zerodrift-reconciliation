# AI Finance Controller

An AI-assisted reconciliation engine that matches orders, gateway settlements, and bank statements ,and is honest about the cases it can't confidently resolve, instead of guessing.

The full scorecard (deterministic results plus the agent's min/median/max across live runs) goes here once Layer 9 wraps and `evaluate.py` has run against the frozen dataset. Until then, this section is a running log of real decisions made while building it ,what changed, why, and what it means for anyone rerunning this.

## A couple of things worth knowing before you dig in

**The agent was built and tested on an open-source model, not Claude, here's why.** Claude is the model this is actually designed for; the only Razorpay-internal fact this project leans on is that Agent Studio runs on Anthropic's Claude Agent SDK, and that's the track this build targets. But iterating on an agent means a lot of calls ,many records, retries, repeated test runs while things are still being debugged ,and that volume doesn't fit inside free Anthropic Console credits. So development runs on Groq's free tier instead, using `openai/gpt-oss-120b`.

That specific model wasn't the original plan, either. The intent going in was Llama 3.3 70B, but by the time this was built, Groq had quietly dropped it from the models their API actually serves ,it just wasn't in the list anymore when checked. `gpt-oss-120b` was the largest open-weights model still available on the key, so that's what ended up carrying the actual development and testing.

None of this is a permanent architectural choice ,swapping back to Claude for production is a one-line change in `src/agent/graph.py::_build_model()`.

**The agent's evaluation numbers are being gathered across several days, not one sitting ,and that's deliberate, not a shortcut.** A full evaluation run means calling the live agent on every record that needs real judgment (37 of them, per the frozen dataset), and doing that three separate times so the reported result is an honest min/median/max rather than one lucky run dressed up as three. Groq's free tier caps out at 200,000 tokens a day, and three full runs need roughly three times that. So the three run logs -> `data/agent_runs/<seed>_1.jsonl`, `_2.jsonl`, `_3.jsonl` -> genuinely have timestamps days apart. That's the token budget talking, not neglect. The full reasoning is written up in `docs/plan.md`'s Layer 8 section if you want the details.
