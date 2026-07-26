#!/bin/bash
# Run the airline task set through the gateway twice — code_mode OFF then ON —
# flipping only the product setting between runs. Restarts the airline tool DB
# clean before each config so the two runs start from the same state.
set -uo pipefail
cd "$(dirname "$0")/.."
export ANTHROPIC_API_KEY=$(python3 -c "print([l.split('=',1)[1].strip().strip('\"') for l in open('$HOME/.avp/demo.env') if 'ANTHROPIC_API_KEY' in l][0])")
N=${N:-10}
AGENT=${AGENT:-claude-sonnet-5}

psql_loco(){ podman exec -e PGPASSWORD=loco portofcontext-postgres psql -U loco -d portal_development -tA -c "$1"; }
set_cm(){ psql_loco "UPDATE portal.orgs SET metadata = COALESCE(metadata,'{}'::jsonb) || '{\"code_mode\":$1}'::jsonb WHERE id='00000000-0000-0000-0000-000000000001';" >/dev/null; echo "  code_mode set to $1"; }
restart_airline(){
  pkill -f airline_mcp.py 2>/dev/null || true; sleep 1
  nohup .venv/bin/python airline_mcp.py >/tmp/airline_mcp.log 2>&1 &
  sleep 2; echo "  airline tool server restarted (clean DB)"
}

echo "===== CONFIG 1: code_mode OFF ====="
restart_airline; set_cm false
N=$N BENCH_N=$N BENCH_CONFIG=code_mode_off BENCH_AGENT_MODEL=$AGENT .venv/bin/python -u gateway_bench/run_bench.py

echo; echo "===== CONFIG 2: code_mode ON ====="
restart_airline; set_cm true
N=$N BENCH_N=$N BENCH_CONFIG=code_mode_on BENCH_AGENT_MODEL=$AGENT .venv/bin/python -u gateway_bench/run_bench.py

echo; echo "===== leaving code_mode OFF (baseline) ====="
set_cm false
echo "ALLDONE"
