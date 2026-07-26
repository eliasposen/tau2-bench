"""Import the tau-bench airline domain into the product's context graph.

Maps airline entities onto the context-graph kinds and emits SQL to bulk-load
portal.context_entities / portal.context_relations for the test org. Airports,
flights, and reservations become `resource`; users become `person`. The flight
network (airport <- flight -> airport) plus user <- reservation -> flight is the
navigable structure the context map surfaces.

Deterministic UUIDs (uuid5) so relations link and re-import is idempotent.
"""

import json
import pathlib
import sys
import uuid

ORG = "00000000-0000-0000-0000-000000000001"
NS = uuid.UUID("00000000-0000-0000-0000-0000000a1a1a")  # airline namespace
DB = str(pathlib.Path(__file__).resolve().parents[1] / "data/tau2/domains/airline/db.json")


def eid(key: str) -> str:
    return str(uuid.uuid5(NS, key))


def sq(s: str) -> str:
    return "'" + str(s).replace("'", "''") + "'"


def main() -> None:
    db = json.load(open(DB))
    flights, users, reservations = db["flights"], db["users"], db["reservations"]

    ents: list[tuple[str, str, str, str]] = []  # (id, kind, name, summary)
    rels: list[tuple[str, str, str]] = []  # (kind, from_id, to_id)
    seen_e: set[str] = set()

    def add_e(key, kind, name, summary):
        i = eid(key)
        if i not in seen_e:
            seen_e.add(i)
            ents.append((i, kind, name, summary))
        return i

    # airports (distinct origins/destinations)
    airports = sorted({f["origin"] for f in flights.values()} | {f["destination"] for f in flights.values()})
    for a in airports:
        add_e(f"airport:{a}", "resource", a, f"Airport {a}.")

    # flights -> related-to origin/destination airports
    for fn, f in flights.items():
        fi = add_e(
            f"flight:{fn}", "resource", fn,
            f"Flight {fn}: {f['origin']} -> {f['destination']}, "
            f"dep {f['scheduled_departure_time_est']} arr {f['scheduled_arrival_time_est']} EST.",
        )
        rels.append(("related-to", fi, eid(f"airport:{f['origin']}")))
        rels.append(("related-to", fi, eid(f"airport:{f['destination']}")))

    # users -> person
    for uid, u in users.items():
        tier = (u.get("membership") or "regular")
        add_e(f"user:{uid}", "person", u["name"] if isinstance(u.get("name"), str)
              else f"{u['name'].get('first_name','')} {u['name'].get('last_name','')}".strip(),
              f"Member ({tier}); user_id {uid}.")

    # reservations -> resource; belongs-to user, related-to flights
    for rid, r in reservations.items():
        ri = add_e(
            f"resv:{rid}", "resource", rid,
            f"Reservation {rid}: {r['origin']}->{r['destination']}, {r.get('cabin','')}, "
            f"{len(r.get('passengers',[]))} pax.",
        )
        if r.get("user_id") and eid(f"user:{r['user_id']}") in seen_e:
            rels.append(("belongs-to", ri, eid(f"user:{r['user_id']}")))
        for leg in r.get("flights", []):
            fn = leg.get("flight_number")
            if fn and eid(f"flight:{fn}") in seen_e:
                rels.append(("related-to", ri, eid(f"flight:{fn}")))

    # emit SQL
    out = ["BEGIN;",
           f"DELETE FROM portal.context_relations WHERE org_id='{ORG}';",
           f"DELETE FROM portal.context_entities WHERE org_id='{ORG}';"]
    # entities
    vals = ",".join(
        f"('{i}','{ORG}','{k}',{sq(n)},{sq(s)})" for i, k, n, s in ents
    )
    out.append(
        "INSERT INTO portal.context_entities (id,org_id,kind,name,summary) VALUES "
        + vals + " ON CONFLICT (id) DO NOTHING;"
    )
    # relations (dedup)
    seen_r = set()
    rvals = []
    for k, a, b in rels:
        key = (k, a, b)
        if key in seen_r:
            continue
        seen_r.add(key)
        rvals.append(f"('{ORG}','{k}','{a}','{b}')")
    out.append(
        "INSERT INTO portal.context_relations (org_id,kind,from_id,to_id) VALUES "
        + ",".join(rvals) + ";"
    )
    out.append("COMMIT;")
    sys.stdout.write("\n".join(out))
    sys.stderr.write(f"entities={len(ents)} relations={len(rvals)} airports={len(airports)}\n")


if __name__ == "__main__":
    main()
