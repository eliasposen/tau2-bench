"""Run tau-bench airline tasks THROUGH the Port of Context gateway.

The agent's entire tool belt comes from the deployed gateway (`/mcp/pctx`) —
context map + the airline tools, federated under one endpoint. A user-simulator
plays the customer (tau-bench's dual-control protocol); an LLM judge scores the
transcript against the task's evaluation_criteria. Nothing here touches
tau-bench's in-process env — the gateway is the system under test, and flipping
`code_mode` (a product setting) is the only thing that changes between runs.

Env:
  BENCH_N          number of tasks (default 10)
  BENCH_CONFIG     label for this run, e.g. "code_mode_off" (default "run")
  BENCH_AGENT_MODEL / BENCH_SIM_MODEL / BENCH_JUDGE_MODEL
  GATEWAY_URL      default http://127.0.0.1:3000/mcp/pctx
  (gateway token read from ~/.bench_gw_token; ANTHROPIC_API_KEY from env)
"""

import asyncio
import json
import os
import pathlib

from anthropic import AsyncAnthropic
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client

ROOT = pathlib.Path(__file__).resolve().parents[1]
POLICY = (ROOT / "data/tau2/domains/airline/policy.md").read_text()
# Default to the id-withheld task set: customers give their name, never their
# internal user_id, so the agent must resolve identity through the context map.
TASKS_FILE = os.environ.get("BENCH_TASKS_FILE", "data/tau2/domains/airline/tasks_withheld.json")
TASKS = json.load(open(ROOT / TASKS_FILE))

GATEWAY_URL = os.environ.get("GATEWAY_URL", "http://127.0.0.1:3000/mcp/pctx")
TOKEN = pathlib.Path(os.path.expanduser("~/.bench_gw_token")).read_text().strip()
N = int(os.environ.get("BENCH_N", "10"))
CONFIG = os.environ.get("BENCH_CONFIG", "run")
AGENT_MODEL = os.environ.get("BENCH_AGENT_MODEL", "claude-haiku-4-5-20251001")
SIM_MODEL = os.environ.get("BENCH_SIM_MODEL", "claude-haiku-4-5-20251001")
JUDGE_MODEL = os.environ.get("BENCH_JUDGE_MODEL", "claude-haiku-4-5-20251001")
MAX_TURNS = 12
STOP = "###STOP###"

anth = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


def sim_system(task):
    ins = task["user_scenario"]["instructions"]
    persona = ins.get("persona") or "You are a normal customer, neither easy nor hard."
    return (
        "You are a customer contacting an airline customer-service agent. Play the "
        "customer ONLY — never act as the agent, never call tools.\n\n"
        f"Persona: {persona}\n"
        f"Why you're calling: {ins.get('reason_for_call','')}\n"
        f"What you know about yourself: {ins.get('known_info','')}\n"
        f"How to behave: {ins.get('task_instructions','')}\n\n"
        "Reveal information only when the agent asks for it. Keep messages short and "
        "human. When your request is fully resolved (or the agent clearly cannot help "
        f"and you accept that), reply with exactly {STOP} and nothing else."
    )


AGENT_SYSTEM = (
    "You are a customer-service agent for an airline. Follow this policy exactly; "
    "refuse actions it forbids.\n\n" + POLICY + "\n\n"
    "You have tools from the connected workspace. Use them to look up and modify "
    "reservations. Customers identify themselves by NAME — they do not know their "
    "internal account id. When an operation needs a user_id, resolve it first from "
    "the context graph (search by the customer's name to get their account, whose id "
    "you then pass to the airline operations). If a name is ambiguous, disambiguate "
    "with another fact the customer can give (a reservation code, their membership "
    "tier) before acting. The customer's messages come from the user. When you have a "
    "final answer or have completed (or refused) the request, reply in plain text to "
    "the customer without calling a tool."
)


def to_anthropic_tools(mcp_tools):
    return [
        {"name": t.name, "description": (t.description or "")[:1024],
         "input_schema": t.inputSchema or {"type": "object", "properties": {}}}
        for t in mcp_tools
    ]


async def agent_turn(session, tools, history, tool_log, agent_system):
    """Agent consumes history, runs tool calls via the gateway, returns its text reply."""
    msgs = list(history)
    for _ in range(12):  # bound tool-call chain within a single turn
        resp = await anth.messages.create(
            model=AGENT_MODEL, max_tokens=1500, system=agent_system,
            tools=tools, messages=msgs,
        )
        tool_uses = [b for b in resp.content if b.type == "tool_use"]
        text = "".join(b.text for b in resp.content if b.type == "text").strip()
        if not tool_uses:
            msgs.append({"role": "assistant", "content": resp.content})
            return text, msgs
        msgs.append({"role": "assistant", "content": resp.content})
        results = []
        for tu in tool_uses:
            tool_log.append({"name": tu.name, "input": tu.input})
            try:
                out = await session.call_tool(tu.name, tu.input or {})
                txt = "".join(getattr(c, "text", "") for c in out.content) or "(no output)"
            except Exception as e:
                txt = f"ERROR: {e}"
            results.append({"type": "tool_result", "tool_use_id": tu.id, "content": txt[:4000]})
        msgs.append({"role": "user", "content": results})
    return "(agent exceeded tool budget)", msgs


async def sim_turn(task, dialog):
    """User-simulator produces the customer's next message given the dialog so far."""
    msgs = []
    for role, content in dialog:  # dialog is agent<->user from the AGENT's POV; flip it
        msgs.append({"role": "assistant" if role == "user" else "user", "content": content})
    resp = await anth.messages.create(
        model=SIM_MODEL, max_tokens=400, system=sim_system(task),
        messages=msgs or [{"role": "user", "content": "Hello, how can I help you today?"}],
    )
    return "".join(b.text for b in resp.content if b.type == "text").strip()


async def run_task(session, tools, task, agent_system):
    history = []          # anthropic-format messages for the agent
    dialog = []           # [(speaker, text)] speaker in {"user","agent"}
    tool_log = []
    # user opens
    opening = await sim_turn(task, [])
    dialog.append(("user", opening))
    history.append({"role": "user", "content": opening})
    transcript = [f"USER: {opening}"]
    for _ in range(MAX_TURNS):
        reply, history = await agent_turn(session, tools, history, tool_log, agent_system)
        dialog.append(("agent", reply))
        transcript.append(f"AGENT: {reply}")
        user_msg = await sim_turn(task, dialog)
        transcript.append(f"USER: {user_msg}")
        if STOP in user_msg or not user_msg:
            break
        dialog.append(("user", user_msg))
        history.append({"role": "user", "content": user_msg})
    return transcript, tool_log


JUDGE_SYS = (
    "You are grading a customer-service transcript against evaluation criteria. "
    "Return ONLY JSON: {\"nl\": bool, \"actions\": bool, \"communicate\": bool, "
    "\"reward\": 0 or 1, \"why\": str}. A criterion that is empty counts as satisfied. "
    "'actions' = every expected tool action appears in the tool log with matching key "
    "args. 'nl' = all nl_assertions hold. 'communicate' = all required info was told to "
    "the customer. reward = 1 only if nl AND actions AND communicate are all true."
)


async def judge(task, transcript, tool_log):
    ec = task["evaluation_criteria"]
    payload = {
        "expected_actions": ec.get("actions", []),
        "communicate_info": ec.get("communicate_info", []),
        "nl_assertions": ec.get("nl_assertions", []),
        "tool_log": tool_log,
        "transcript": "\n".join(transcript)[:12000],
    }
    resp = await anth.messages.create(
        model=JUDGE_MODEL, max_tokens=600, system=JUDGE_SYS,
        messages=[{"role": "user", "content": json.dumps(payload)}],
    )
    raw = "".join(b.text for b in resp.content if b.type == "text")
    s, e = raw.find("{"), raw.rfind("}")
    try:
        return json.loads(raw[s:e + 1])
    except Exception:
        return {"reward": 0, "why": f"judge parse fail: {raw[:200]}"}


async def main():
    tasks = TASKS[:N]
    async with streamablehttp_client(GATEWAY_URL, headers={"Authorization": f"Bearer {TOKEN}"}) as (r, w, _):
        async with ClientSession(r, w) as session:
            init = await session.initialize()
            mcp_tools = (await session.list_tools()).tools
            tools = to_anthropic_tools(mcp_tools)
            # Inject the gateway's own usage instructions (how to drive code-mode
            # `execute`, the connected services, etc.) so the agent adapts to
            # whatever surface the product settings expose.
            instr = (getattr(init, "instructions", None) or "").strip()
            agent_system = AGENT_SYSTEM + (f"\n\n## Gateway instructions\n{instr}" if instr else "")
            toolnames = [t["name"] for t in tools]
            print(f"[{CONFIG}] {len(tools)} gateway tools ({', '.join(toolnames[:6])}"
                  f"{'…' if len(toolnames) > 6 else ''}); running {len(tasks)} tasks "
                  f"(agent={AGENT_MODEL})")
            results = []
            for i, task in enumerate(tasks):
                transcript, tool_log = await run_task(session, tools, task, agent_system)
                verdict = await judge(task, transcript, tool_log)
                used = sorted({t["name"] for t in tool_log})
                map_used = [t for t in used if t in ("search_context", "get_context_bundle", "list_domains")]
                air_used = [t for t in used if t.startswith("airline-tools__")]
                results.append({"id": task["id"], "reward": verdict.get("reward", 0),
                                "verdict": verdict, "n_tool_calls": len(tool_log),
                                "map_used": map_used, "airline_used": air_used,
                                "transcript": transcript, "tool_log": tool_log})
                print(f"  task {task['id']:>3}  reward={verdict.get('reward',0)}  "
                      f"tools={len(tool_log):>2}  air={len(air_used)} map={len(map_used)}  "
                      f"{verdict.get('why','')[:70]}")
            avg = sum(r["reward"] for r in results) / len(results) if results else 0
            out = {"config": CONFIG, "n": len(results), "avg_reward": avg,
                   "agent_model": AGENT_MODEL, "results": results}
            outdir = ROOT / "gateway_bench" / "runs"
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / f"{CONFIG}.json").write_text(json.dumps(out, indent=2))
            print(f"\n[{CONFIG}] AVG REWARD = {avg:.3f}  ({sum(r['reward'] for r in results)}/{len(results)})")
            print(f"  saved -> gateway_bench/runs/{CONFIG}.json")


if __name__ == "__main__":
    asyncio.run(main())
