# gateway_bench — tau-bench through the Port of Context gateway

Runs τ²-bench **airline** tasks against the **deployed pctx gateway** (`/mcp/pctx`),
where tool exposure is controlled by product settings (connected MCP servers,
`code_mode`). The gateway is the system under test; τ²-bench supplies the tasks
and the dual-control (agent ↔ user-simulator) protocol; an LLM judge scores each
transcript against the task's `evaluation_criteria`.

Design rationale: [`DESIGN.md`](./DESIGN.md).

## What makes the map load-bearing

The raw airline tasks hand the agent the customer's internal `user_id`, so entity
resolution is free and the context map is never needed. [`withhold_ids.py`](./withhold_ids.py)
strips the id from every task's customer brief, leaving only the name. Now the
agent *must* resolve name → account id through the context map before it can use
any airline tool — the realistic case (customers know their name, not their id).
This id-withheld set is the default the runner loads.

## Pieces

| File | Role |
|---|---|
| `../airline_mcp.py` | Exposes τ²-bench's 14 airline tools as an MCP server (`:8811`), **with output schemas** derived from the domain's Pydantic return types. |
| `import_airline.py` | Loads the airline graph (flights/airports/reservations/users) into the org's context map (`context_entities`/`_relations`). |
| `setup_gateway.sh` | Idempotently wires it onto the gateway: mints a token, registers `airline-tools`, vaults a connection, imports the map. |
| `withhold_ids.py` | Generates `tasks_withheld.json` — the id-stripped task set. |
| `run_bench.py` | The runner: agent ↔ user-sim through `/mcp/pctx`, then judge. Adapts to whatever surface the gateway exposes (direct tools *or* code-mode `execute_tool`). |
| `compare.sh` | Runs the set with `code_mode` off then on. |
| `probe.py` / `verify_fix.py` | List the gateway's tools / verify context tools return `structured_content`. |

## Run it

```bash
# prerequisites: compose stack up (portal/agentgateway/postgres);
# a code-mode executor on :8080 (`pctx start --host 0.0.0.0 --port 8080`);
# ANTHROPIC_API_KEY in the environment.

python airline_mcp.py &                       # airline tools on :8811
bash gateway_bench/setup_gateway.sh           # wire onto the gateway + import map
python gateway_bench/withhold_ids.py          # generate the id-withheld tasks
BENCH_N=10 BENCH_AGENT_MODEL=claude-sonnet-5 \
  python gateway_bench/run_bench.py           # run + judge (writes gateway_bench/runs/<config>.json)
```

Knobs (env): `BENCH_N`, `BENCH_CONFIG`, `BENCH_AGENT_MODEL` / `BENCH_SIM_MODEL` /
`BENCH_JUDGE_MODEL`, `BENCH_TASKS_FILE`, `GATEWAY_URL`. The gateway token is read
from `~/.bench_gw_token` (written by `setup_gateway.sh`).

## Results (10 id-withheld airline tasks, sonnet agent)

| Gateway config | Reward | Notes |
|---|---|---|
| ids given (map redundant) | 0.60 | map used 0/10 — agent goes straight to the id |
| ids withheld, direct federation | 0.50 | map used 10/10 — `search_context` → account id → airline tools |
| ids withheld, **code mode** | 0.50 | `get_context` (map) → `get_tool_details` (typed) → `execute_tool` (typed TS calls) |

The map absorbs the identity-resolution burden the raw tools can't do at all;
code mode matches direct federation once the tools carry output schemas (the typed
TS surface is what lets the agent write `user.reservations` etc.).
