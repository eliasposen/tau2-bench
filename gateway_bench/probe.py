import asyncio, os
from mcp.client.streamable_http import streamablehttp_client
from mcp import ClientSession

TOK = open(os.path.expanduser("~/.bench_gw_token")).read().strip()
URL = "http://127.0.0.1:3000/mcp/pctx"

async def main():
    async with streamablehttp_client(URL, headers={"Authorization": f"Bearer {TOK}"}) as (r, w, _):
        async with ClientSession(r, w) as s:
            await s.initialize()
            tools = (await s.list_tools()).tools
            print(f"{len(tools)} tools:")
            for t in tools:
                print("  -", t.name)

asyncio.run(main())
