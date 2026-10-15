# Quick take from the repo before the task

**What's actually missing (the real gap):** every "agent" you've built — DB agent, mail agent, math/file agent — does exactly one tool call per turn. You call `llm.invoke()`, check `if tool_calls:`, run the one tool via manual `if/elif`, print the result, done. There's no loop where the tool's output goes *back* to the model so it can decide what to do next. You've also never used `with_structured_output` (you're just reading raw `.tool_calls[0]['args']` dicts, no Pydantic validation), never chained multiple tools where one tool's output feeds the next tool's input, and never used `AgentExecutor` / `create_tool_calling_agent`. That's exactly where LangChain stops being "nicer syntax" and starts being "actually saves you from writing a state machine."

**Problem:**
You're prototyping a backend helper for an internal ops tool. Someone types a messy natural-language purchase request like:

> "I want to order 3 kg of coffee beans priced at $12.50 per pound, convert that to INR, add 8% tax, and tell me if it fits in a ₹5000 budget."

The system needs to figure out, on its own, which conversions and calculations are needed and in what order, call the right tools in sequence (using earlier results as inputs to later tools), and hand back a clean, validated final answer — not a wall of raw text.

**Your role:**
Backend engineer building the orchestration layer for this request-processing agent.

**Goal:**
Build a small LangChain agent that resolves a shopping/budget request requiring **multiple sequential tool calls**, where later tool calls depend on earlier tool outputs, and returns a **structured (Pydantic-validated) final answer**.

**Requirements:**
-Define at least 3 tools with `@tool`: a unit converter (kg↔lb, etc.), a currency converter (use a fixed/mock exchange rate dict — no need for a live API), and a tax/budget calculator.
-Use `create_tool_calling_agent` + `AgentExecutor` (or equivalent current LangChain agent construct) instead of manually reading `.tool_calls[0]` — let LangChain manage the "call tool → feed result back → decide next step" loop.
-Define a Pydantic model for the final answer (e.g. `total_cost_inr: float`, `within_budget: bool`, `steps: list[str]`) and get the agent's final response validated against it — not raw string parsing.
-Test with at least 2 different phrasings/requests, including one where the budget is exceeded and one where it isn't.

**Constraints:**
-No live currency API — hardcode a small exchange-rate dict, that's fine and realistic for a prototype.
-Keep it to a single file or two. This is not a multi-service project.
-You may use OpenAI or Gemini as the underlying LLM — whichever you have keys for.

**Expected output:**
A working script where you can hand it a natural-language request and it prints (or returns) the validated structured result, plus a print/log of which tools were called in what order (so you can *see* the orchestration happening).

**Success criteria:**
-The agent correctly chains tool calls (conversion result flows into the tax/budget tool) without you manually wiring the sequence with if/elif.
-Final output is a validated Pydantic object, not a raw string you're regex-parsing.
-If you swap the LLM provider (OpenAI ↔ Gemini), the orchestration code doesn't change — only the model init line does.
-You can explain, in your own words, at which exact point this would have gotten painful with raw SDK tool-calling (this is the part I'll ask about when you bring it back).

**What you're allowed to use:**
-Already known: `@tool`, `bind_tools`, prompt templates, reading `.env` keys, basic Python.
-New concepts to look up in docs (this is expected, not a gap): `create_tool_calling_agent`, `AgentExecutor`, `with_structured_output` (or `PydanticOutputParser` if your model doesn't support native structured output well), and how `agent_scratchpad` works in the agent prompt.
-Official LangChain docs/how-to guides are fair game — reading docs while building is the point, not cheating.

Bring back whatever you get working, even half-broken — that's more useful to review than nothing.

agent response: total_cost_inr=₹5340.00 | within_budget=False | steps=['converted 3kg→lb', 'converted USD→INR', 'applied 8% tax']
