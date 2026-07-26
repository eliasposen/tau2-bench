"""Expose the tau-bench airline domain's tools as a streamable-HTTP MCP server,
so the product gateway can federate them next to the airline context map.

The 15 airline tools operate on the in-process airline DB (the same state the
tau-bench scorer reads). This wraps each with its real schema and dispatches to
the toolkit — turning "a benchmark's tools" into "connected MCP tools behind the
gateway", which is what the gateway is for.
"""

import contextlib
import inspect
import json

import mcp.types as types
import uvicorn
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from pydantic import BaseModel, TypeAdapter
from starlette.applications import Starlette
from starlette.routing import Mount

from tau2.domains.airline.environment import get_environment

ENV = get_environment()
TOOLKIT = ENV.tools
TOOLS = TOOLKIT.get_tools()  # name -> Tool (carries openai_schema)


def _build_output_schema(method):
    """Derive an MCP outputSchema + serializer from a tool's return annotation.

    tau-bench tools return Pydantic models / lists of them / str. MCP requires
    structuredContent to be a JSON *object*, so non-object returns (str, list,
    list[tuple]) are wrapped under `result`. Returns (adapter, schema, is_object)
    or (None, None, False) when there's no annotation to type.
    """
    anno = inspect.signature(method).return_annotation
    if anno is inspect.Signature.empty:
        return None, None, False
    adapter = TypeAdapter(anno)
    schema = adapter.json_schema()
    defs = schema.pop("$defs", None)  # hoist to whichever root we return
    is_object = schema.get("type") == "object"
    if is_object:
        out = schema
    else:
        out = {
            "type": "object",
            "properties": {"result": schema},
            "required": ["result"],
            "additionalProperties": False,
        }
    if defs:
        out["$defs"] = defs
    return adapter, out, is_object


# name -> (adapter, output_schema, is_object)
SCHEMAS = {name: _build_output_schema(getattr(TOOLKIT, name)) for name in TOOLS}

server = Server("airline")


def _to_text(result) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, BaseModel):
        return result.model_dump_json()
    if isinstance(result, list):
        return json.dumps([r.model_dump() if isinstance(r, BaseModel) else r for r in result], default=str)
    try:
        return json.dumps(result, default=str)
    except Exception:
        return str(result)


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    out = []
    for name, t in TOOLS.items():
        fn = t.openai_schema["function"]
        _, schema, _ = SCHEMAS[name]
        out.append(
            types.Tool(
                name=name,
                description=fn.get("description", ""),
                inputSchema=fn.get("parameters") or {"type": "object", "properties": {}},
                outputSchema=schema,
            )
        )
    return out


@server.call_tool()
async def call_tool(name: str, arguments: dict):
    method = getattr(TOOLKIT, name, None)
    if method is None:
        # is_error result is returned as-is (no outputSchema validation).
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Error: no tool {name!r}")],
            isError=True,
        )
    try:
        result = method(**(arguments or {}))
    except Exception as e:  # surface tool errors like the domain does
        return types.CallToolResult(
            content=[types.TextContent(type="text", text=f"Error: {e}")],
            isError=True,
        )
    content = [types.TextContent(type="text", text=_to_text(result))]
    adapter, _, is_object = SCHEMAS[name]
    if adapter is None:
        return content
    value = adapter.dump_python(result, mode="json")
    structured = value if is_object else {"result": value}
    # (content, structured): the server validates `structured` against the
    # tool's outputSchema and errors loudly if they diverge.
    return content, structured


session_manager = StreamableHTTPSessionManager(app=server, json_response=False, stateless=True)


async def handle(scope, receive, send):
    await session_manager.handle_request(scope, receive, send)


@contextlib.asynccontextmanager
async def lifespan(app):
    async with session_manager.run():
        yield


app = Starlette(routes=[Mount("/mcp", app=handle)], lifespan=lifespan)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8811, log_level="warning")
