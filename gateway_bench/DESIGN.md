# Running tau-bench through the Port of Context gateway

**Goal.** Run tau-bench tasks (start with `airline`) through the **deployed
gateway**, where **tool exposure is controlled by product settings** (code
mode, connected MCP servers). Measure how gateway configuration changes agent
performance. The gateway is the system under test — not a client library.

## Not the path: Elias's in-process harness (reference only)

`src/tau2/agent/llm_pctx_agent.py` runs tau-bench's tools in-process through
`pctx_client`, with `PCTX_MODE` (code / fs / python / direct). Keep it as a
**reference** — its code-mode and disclosure *prompt patterns* are worked out
and worth reusing. But it is not the target, because:

- it exercises the pctx **client**, not the **deployed gateway**;
- tool exposure is set by an **env var**, not by **product settings**.

## Shape: lift the questions out, run them through the gateway

Three parts, deliberately decoupled:

1. **Tasks (data, from tau-bench).** Each task is a question + eval criteria
   (`actions` + `nl_assertions`) + the domain DB seed. These are just JSON —
   export them and carry them to any harness.
2. **Tools on the gateway.** `airline_mcp.py` exposes tau-bench's airline
   toolkit as an MCP server; register it via `org_mcp_servers`. The gateway
   then serves the airline tools **and** the context map, and applies
   `code_mode` (and future disclosure settings) per **product config**.
3. **Harness.** Runs an agent that connects **only** to `/mcp/pctx`, feeds it
   the task instruction (add a user-simulator later for multi-turn), and
   captures the trajectory. **avp works here** (load the tau-bench questions as
   avp eval items), or a ~100-line runner.

## The scoring problem, and the fix

tau-bench scores by reading its **in-process DB** after a run. Through the
deployed gateway that DB lives in `airline_mcp.py`'s process, not the harness —
so tau-bench's native scorer can't see it. Don't fight it; score **outside**
tau-bench:

- **Answer** → LLM judge over the task's `nl_assertions`.
- **Actions** → check the trajectory's tool calls against the task's expected
  `actions` (names + key args).

Optional, to recover full write-task rigor **through the gateway**: give
`airline_mcp.py` two endpoints — `reset(seed)` before each task and
`get_state()` after — so the scorer reads the true post-run DB. This restores
tau-bench's state assertions without ever leaving the gateway path.

## The experiment = product settings, flipped between runs

Each run is a **gateway configuration**, changed through product settings, not
code:

- `code_mode` on / off (org feature flag)
- which tools/connectors are exposed (`org_mcp_servers`)
- (future) disclosure mode, tool descriptions

Run the same task set under each config; compare score + tool-call efficiency.
That comparison is the product signal — "does this gateway setting help the
agent."

## First slice

1. `airline_mcp.py`: add `/reset` + `/state` (per-task DB).
2. Export N airline tasks to portable JSON (instruction, expected `actions`,
   `nl_assertions`).
3. Harness: for each task × each gateway config → run agent through
   `/mcp/pctx` → judge → record. avp eval items are one concrete way to hold
   the tasks + judge.
4. First comparison: `code_mode` on vs off, airline, N tasks.

## What's already standing

- Gateway up on `:3000` (`/mcp/pctx`).
- `airline_mcp.py` up on `:8811`, registered as `airline-tools` on the gateway.
- Airline map imported into the graph (org `…0001`).
- Product patches supporting local runs: `compose.yaml` (code-mode executor),
  `provider_resolver.rs` (allow registering a local MCP server) — formalize
  these into real settings.
