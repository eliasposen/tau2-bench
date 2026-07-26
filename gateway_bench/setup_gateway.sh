#!/bin/bash
# Reproducibly wire the airline domain onto the deployed Port of Context gateway.
# Idempotent: safe to re-run. Assumes the compose stack (portal/agentgateway/
# postgres) is up and airline_mcp.py is (or will be) listening on :8811.
#
#   1. mint a gateway token for the agent  -> ~/.bench_gw_token + gateway_tokens
#   2. register airline_mcp.py as a custom MCP server (org_mcp_servers)
#   3. vault a dummy api-key connection so the gateway federates it
#   4. import the airline graph as the org's context map (context_entities/relations)
#
# All DB writes go through the `loco` superuser to bypass row-level security.
set -uo pipefail
cd "$(dirname "$0")/.."
ORG=00000000-0000-0000-0000-000000000001
USER=00000000-0000-0000-0000-000000000002
PY=.venv/bin/python
psql_loco(){ podman exec -e PGPASSWORD=loco portofcontext-postgres psql -U loco -d portal_development -tA -c "$1"; }

echo "1. mint gateway token"
TOK=$($PY -c "import os,base64;print(base64.urlsafe_b64encode(os.urandom(32)).rstrip(b'=').decode())")
HASH=$($PY -c "import hashlib;print(hashlib.sha256('$TOK'.encode()).hexdigest())")
printf '%s' "$TOK" > "$HOME/.bench_gw_token"
psql_loco "DELETE FROM portal.gateway_tokens WHERE label='bench';
 INSERT INTO portal.gateway_tokens (id,org_id,user_id,token_hash,label,principal_type,created_at)
 VALUES (gen_random_uuid(),'$ORG','$USER','$HASH','bench','human',now());" >/dev/null
echo "   token -> ~/.bench_gw_token (${TOK:0:8}…)"

echo "2. register airline_mcp.py as org MCP server"
psql_loco "DELETE FROM portal.org_mcp_servers WHERE slug='airline-tools';
 INSERT INTO portal.org_mcp_servers (id,org_id,slug,name,mcp_url,enabled,created_by,created_at,auth_type,header_name,category)
 VALUES (gen_random_uuid(),'$ORG','airline-tools','Airline Tools','http://host.docker.internal:8811/mcp/',true,'$USER',now(),'api_key','authorization','custom');" >/dev/null

echo "3. vault a dummy connection (airline_mcp ignores the token; gateway needs a valid ciphertext)"
ENCKEY=$(podman exec portofcontext_portal_1 printenv PORTAL_TOKEN_ENC_KEY)
CIPHER=$($PY -c "
import os,base64
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
k=base64.b64decode('$ENCKEY'); n=os.urandom(12)
print(base64.b64encode(n+AESGCM(k).encrypt(n,b'ignored',None)).decode())")
psql_loco "DELETE FROM portal.oauth_connections WHERE provider='airline-tools';
 INSERT INTO portal.oauth_connections (id,org_id,user_id,provider,provider_account_id,access_token,created_at,updated_at,needs_reauth)
 VALUES (gen_random_uuid(),'$ORG','$USER','airline-tools','bench','$CIPHER',now(),now(),false);" >/dev/null

echo "4. import airline graph as the context map"
$PY gateway_bench/import_airline.py > /tmp/airline_map.sql
podman exec -e PGPASSWORD=loco -i portofcontext-postgres psql -U loco -d portal_development < /tmp/airline_map.sql >/dev/null
N=$(psql_loco "SELECT count(*) FROM portal.context_entities WHERE org_id='$ORG';")
echo "   context_entities: $N"

echo "done. verify with: $PY gateway_bench/probe.py"
