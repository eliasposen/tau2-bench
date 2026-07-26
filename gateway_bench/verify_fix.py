"""Call search_context through the gateway with a STOCK mcp client (no validation
skip). Before the fix this raised 'has an output schema but did not return
structured content'; after it, it returns cleanly with structured_content."""
import asyncio, os, sys
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession
TOK = open(os.path.expanduser("~/.bench_gw_token")).read().strip()
async def main():
    async with streamablehttp_client("http://127.0.0.1:3000/mcp/pctx",
            headers={"Authorization": f"Bearer {TOK}"}) as (r,w,_):
        async with ClientSession(r,w) as s:
            await s.initialize()
            out = await s.call_tool("search_context", {"query": "Emma Kim"})
            sc = out.structuredContent
            txt = "".join(getattr(c,"text","") for c in out.content)
            print("OK: search_context returned",
                  "structured_content" if sc else "NO structured_content",
                  f"({len(txt)} text chars)")
try:
    asyncio.run(main()); print("FIX_VERIFIED")
except Exception as e:
    print("STILL_BROKEN:", type(e).__name__, str(e)[:120]); sys.exit(1)
