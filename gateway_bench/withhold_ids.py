"""Withhold internal user_ids from the airline tasks — permanently.

A real customer says "I'm Emma Kim", not "I'm emma_kim_9957". The raw tasks hand
the agent the internal id, so entity resolution is free and the context map is
never needed. This strips the user_id from the customer's brief and guarantees
the customer's NAME is present instead, so the agent must resolve name -> id via
the map (search_context) before it can use the airline tools. This is THE task
set, not a variant.

Only the customer-facing instruction text is touched; evaluation_criteria
(expected actions/args) are left exactly as tau-bench defines them.
"""

import json
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
AIR = ROOT / "data/tau2/domains/airline"
UID = re.compile(r"[a-z]+_[a-z]+_\d+")


def full_name(user) -> str:
    n = user["name"]
    return f"{n['first_name']} {n['last_name']}" if isinstance(n, dict) else str(n)


def main() -> None:
    tasks = json.load(open(AIR / "tasks.json"))
    users = json.load(open(AIR / "db.json"))["users"]
    out = []
    for t in tasks:
        ins = t["user_scenario"]["instructions"]
        ki = ins.get("known_info") or ""
        # the user_id: prefer the one the expected actions use, else scan known_info
        uid = None
        for a in t["evaluation_criteria"].get("actions", []):
            for k, v in (a.get("arguments") or {}).items():
                if k in ("user_id", "uid") and isinstance(v, str) and UID.fullmatch(v):
                    uid = v
        if uid is None:
            m = UID.search(ki)
            uid = m.group(0) if m else None
        name = full_name(users[uid]) if uid in users else None

        # rebuild known_info: drop name/user-id lines, keep everything else
        # (confirmation/reservation numbers etc.), then re-assert the name.
        kept = []
        for line in ki.splitlines():
            s = line.strip()
            if not s:
                continue
            low = s.lower()
            if (uid and uid in s) or "user id" in low or "user_id" in low \
                    or low.startswith("you are ") or low.startswith("your name is"):
                continue
            kept.append(s)
        name = name or "the customer"
        new_ki = f"You are {name}." + ("\n" + "\n".join(kept) if kept else "")

        # scrub any stray uid token from the other instruction fields too
        def scrub(x):
            return x.replace(uid, name) if (uid and isinstance(x, str)) else x

        ins["known_info"] = new_ki
        ins["reason_for_call"] = scrub(ins.get("reason_for_call"))
        ins["task_instructions"] = scrub(ins.get("task_instructions"))
        out.append(t)

    dst = AIR / "tasks_withheld.json"
    dst.write_text(json.dumps(out, indent=2))
    print(f"wrote {len(out)} id-withheld tasks -> {dst.relative_to(ROOT)}")
    for t in out[:6]:
        ki = t["user_scenario"]["instructions"]["known_info"].replace("\n", " | ")
        leaked = "  <-- LEAK" if UID.search(ki) else ""
        print(f"  {t['id']:>3}: {ki[:90]}{leaked}")


if __name__ == "__main__":
    main()
