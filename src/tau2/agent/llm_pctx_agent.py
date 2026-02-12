import asyncio
import functools
import json
import uuid
from copy import deepcopy
from datetime import datetime
from typing import Callable, Coroutine

from loguru import logger
from pctx_client import Pctx
from pctx_client import tool as pctx_tool

from tau2.agent.base import ValidAgentInputMessage
from tau2.agent.llm_agent import LLMAgent, LLMAgentState
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
)
from tau2.environment.environment import Environment
from tau2.environment.tool import as_tool
from tau2.utils.utils import get_now


class LLMPctxAgent(LLMAgent):
    def __init__(
        self,
        env: Environment,
        llm: str | None = None,
        llm_args: dict | None = None,
    ):
        self.internal_messages: list[Message] = []
        self.current_execute_callbacks: list[tuple[ToolCall, ToolMessage]] = []
        self.env = env

        # Create a persistent event loop for this agent instance
        self._loop = asyncio.new_event_loop()

        # convert env tools to pctx tools for code-mode registration
        env_tools = list(env.tools.tools.values()) if env.tools else []
        pctx_env_tools = [
            pctx_tool(self._track_pctx_tool(fn), namespace=env.domain_name)
            for fn in env_tools
        ]
        self.pctx = Pctx(tools=pctx_env_tools)
        self.code_mode_fns = self._get_sync_code_mode_fns()
        tau_tools = [as_tool(t) for t in self.code_mode_fns.values()]

        super().__init__(tau_tools, env.get_policy(), llm, llm_args)

    def __del__(self):
        """Clean up the persistent event loop when the agent is deleted."""
        if hasattr(self, "_loop") and not self._loop.is_closed():
            self._loop.close()

    def _track_pctx_tool(self, fn: Callable) -> Callable:
        """
        Returns wrapped callable tracing arguments as tool calls
        without altering the original type signature
        """

        @functools.wraps(fn)
        def tracked(**kwargs):
            tool_call_id = str(uuid.uuid4())
            tool_call = ToolCall(id=tool_call_id, name=fn.__name__, arguments=kwargs)

            logger.debug(f"[PCTX] Env Call - {tool_call.name}\n{tool_call.arguments}")

            result = None
            error = False
            try:
                result = fn(**kwargs)
            except Exception as e:
                result = f"Error: {e}"
                error = True

            content = self.env.to_json_str(result)

            logger.debug(
                f"[PCTX] Env Response - {tool_call.name} (error={error})\n{content}"
            )

            tool_msg = ToolMessage(
                id=tool_call_id,
                role="tool",
                content=content,
                error=error,
            )
            self.current_execute_callbacks.append((tool_call, tool_msg))

            return result

        return tracked

    def _handle_pctx_tool_call(self, tool_call: ToolCall) -> ToolMessage:
        error = False
        logger.debug(
            f"[PCTX] Call - {tool_call.name}\n{tool_call.arguments.get('functions', tool_call.arguments.get('code', tool_call.arguments))}"
        )
        try:
            resp = self.code_mode_fns[tool_call.name](**tool_call.arguments)
        except Exception as e:
            resp = f"Error: {e}"
            error = True

        if tool_call.name == "pctx_execute":
            logger.debug(f"[PCTX] Response - {tool_call.name} (error={error})\n{resp}")
        else:
            logger.debug(f"[PCTX] Response - {tool_call.name} (error={error})")

        return ToolMessage(
            id=tool_call.id,
            content=json.dumps(resp),
            requestor=tool_call.requestor,
            role="tool",
            error=error,
        )

    def _run_in_loop(self, coro: Coroutine):
        """Run a coroutine in the agent's persistent event loop from sync code."""
        asyncio.set_event_loop(self._loop)
        try:
            return self._loop.run_until_complete(coro)
        finally:
            asyncio.set_event_loop(None)

    def _get_sync_code_mode_fns(self) -> dict[str, Callable]:
        """Get synchronous wrapper functions for pctx code mode tools."""

        def pctx_list_functions() -> str:
            return self._run_in_loop(self.pctx.list_functions()).code

        pctx_list_functions.__doc__ = CODE_MODE_TOOL_DESCRIPTIONS["list_functions"]

        def pctx_get_function_details(functions: list[str]) -> str:
            return self._run_in_loop(self.pctx.get_function_details(functions)).code

        pctx_get_function_details.__doc__ = CODE_MODE_TOOL_DESCRIPTIONS[
            "get_function_details"
        ]

        def pctx_execute(code: str) -> str:
            return self._run_in_loop(self.pctx.execute(code)).markdown()

        pctx_execute.__doc__ = CODE_MODE_TOOL_DESCRIPTIONS["execute"]

        return {
            "pctx_list_functions": pctx_list_functions,
            "pctx_get_function_details": pctx_get_function_details,
            "pctx_execute": pctx_execute,
        }

    def connect(self):
        self._run_in_loop(self.pctx.connect())
        logger.debug(
            f"[PCTX] - connected to server: session_id={self.pctx._session_id}"
        )

    def disconnect(self):
        self._run_in_loop(self.pctx.disconnect())
        logger.debug(f"[PCTX] - disconnected from server")

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        msg, state = super().generate_next_message(message=message, state=state)

        # TODO: track these internal messages
        iteration = 0
        while msg.is_tool_call():
            iteration += 1
            logger.debug(
                f"[PCTX] Tool call iteration {iteration}\n\ttool call(s): {len(msg.tool_calls)}\n\tmessage content: {msg.content}"
            )

            expanded_assistant_msg = deepcopy(msg)
            expanded_assistant_msg.tool_calls = []

            tool_msgs = []
            execute_tool_msgs = []
            for pctx_tool_call in msg.tool_calls:
                self.current_execute_callbacks.clear()

                before_handle = get_now()
                pctx_tool_msg = self._handle_pctx_tool_call(pctx_tool_call)
                pctx_tool_msg.timestamp = before_handle
                tool_msgs.append(pctx_tool_msg)

                expanded_assistant_msg.tool_calls.append(pctx_tool_call)
                expanded_assistant_msg.tool_calls.extend(
                    map(lambda e: e[0], self.current_execute_callbacks)
                )
                execute_tool_msgs.extend(
                    map(lambda e: e[1], self.current_execute_callbacks)
                )

            self.internal_messages.append(expanded_assistant_msg)
            self.internal_messages.extend(tool_msgs)
            self.internal_messages.extend(execute_tool_msgs)

            # Packaging multiple tool messages into a MultiToolMessage
            if len(tool_msgs) > 1:
                logger.debug(
                    f"[PCTX] Packaging {len(tool_msgs)} tool messages into MultiToolMessage"
                )
                next_msg = MultiToolMessage(
                    role="tool",
                    tool_messages=tool_msgs,
                )
            else:
                next_msg = tool_msgs[0]

            # call model again
            logger.debug(f"[PCTX] Calling model again with tool results")
            msg, state = super().generate_next_message(message=next_msg, state=state)

        logger.debug(
            f"[PCTX] Returning final message after {iteration} tool call iteration(s)"
        )
        # final msg will automatically be added to the trajectory so we should avoid
        # adding to self.internal_messages (double counting)
        return msg, state

    def get_internal_messages(self) -> list[Message]:
        return self.internal_messages


CODE_MODE_TOOL_DESCRIPTIONS = {
    ##############
    # list_functions
    ##############
    "list_functions": """ALWAYS USE THIS TOOL FIRST to list all available functions organized by namespace.

WORKFLOW:
1. Start here - Call this tool to see what functions are available
2. Then call get_function_details() for specific functions you need to understand
3. Finally call execute() to run your TypeScript code

This returns function signatures without full details.""",
    "search_functions": """ALWAYS USE THIS TOOL FIRST to find relevant functions.

Arguments:
  query: The search query string to find relevant functions.
  k: The maximum number of top results to return (default: 10).


WORKFLOW:
1. Start here - Call this tool to find suitable functions
2. Then call get_function_details() for specific functions you need to understand
3. Finally call execute() to run your TypeScript code

This returns a list of matching functions.""",
    ##############
    # get_function_details
    ##############
    "get_function_details": """Get detailed information about specific functions you want to use.

WHEN TO USE: After calling list_functions(), use this to learn about parameter types, return values, and usage for specific functions.

REQUIRED FORMAT: Functions must be specified as 'namespace.functionName' (e.g., 'Namespace.apiPostSearch')

This tool is lightweight and only returns details for the functions you request, avoiding unnecessary token usage.
Only request details for functions you actually plan to use in your code.

NOTE ON RETURN TYPES:
- If a function returns Promise<any>, the MCP server didn't provide an output schema
- The actual value is a parsed object (not a string) - access properties directly
- Don't use JSON.parse() on the results - they're already JavaScript objects""",
    ##############
    # execute
    ##############
    "execute": """Execute TypeScript code that calls namespaced functions. USE THIS LAST after list_functions() and get_function_details().

TOKEN USAGE WARNING: This tool could return LARGE responses if your code returns big objects.
To minimize tokens:
- Filter/map/reduce data IN YOUR CODE before returning
- Only return specific fields you need (e.g., return {id: result.id, count: items.length})
- Use console.log() for intermediate results instead of returning everything
- Avoid returning full API responses - extract just what you need

REQUIRED CODE STRUCTURE:
async function run() {
    // Your code here
    // Call namespace.functionName() - MUST include namespace prefix
    // Process data here to minimize return size
    return onlyWhatYouNeed; // Keep this small!
}

IMPORTANT RULES:
- You MUST define a `run()` function
- You MUST NOT call or export any functions from the root of the script, `run()` will be called automatically
- ALWAYS batch multiple tool operations into ONE execute call.
- Functions MUST be called as 'Namespace.functionName' (e.g., 'Notion.apiPostSearch')
- Only functions from list_functions() are available - no fetch(), fs, or other Node/Deno APIs
- Variables don't persist between execute() calls - return or log anything you need later
- Add console.log() statements between API calls to track progress if errors occur
- Code runs in an isolated Deno sandbox with restricted network access

RETURN TYPE NOTE:
- Functions without output schemas show Promise<any> as return type
- The actual runtime value is already a parsed JavaScript object, NOT a JSON string
- Do NOT call JSON.parse() on results - they're already objects
- Access properties directly (e.g., result.data) or inspect with console.log() first
- If you see 'Promise<any>', the structure is unknown - log it to see what's returned""",
}
