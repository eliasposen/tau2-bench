import asyncio
import functools
import json
import os
import uuid
from copy import deepcopy
from typing import Callable, Coroutine

from loguru import logger
from pctx_client import Pctx
from pctx_client import tool as pctx_tool
from pctx_client.tool_descriptions import PRESCRIPTIVE_DESCRIPTIONS

from tau2.agent.base import ValidAgentInputMessage
from tau2.agent.llm_agent import LLMAgent, LLMAgentState, LLMSoloAgent
from tau2.data_model.message import (
    AssistantMessage,
    Message,
    MultiToolMessage,
    ToolCall,
    ToolMessage,
)
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment
from tau2.environment.tool import as_tool
from tau2.utils.utils import get_now

# fs mode addendum
PCTX_FS_ADDENDUM = """

Available functions in `{namespace}` namespace:
{function_list}

Use pctx_execute_typescript to call functions as `await {namespace}.functionName({{ args }})`.

**Communication vs Action:**
- When you need information from the user (dates, preferences, clarifications): ASK them, don't try to figure it out yourself
- When you have specific information: use tools to get details or take actions
- Don't do exhaustive searches - be conversational and gather requirements first

Write operations (cancel/book/update/send_certificate) execute database changes.
Multi-step pattern: gather info → verify policy → execute action → communicate result.""".strip()


class LLMPctxMixin:
    """Mixin providing pctx code/fs execution capabilities to LLM agents.

    Intended for use as a left-hand base alongside LLMAgent or LLMSoloAgent:

        class LLMPctxAgent(LLMPctxMixin, LLMAgent): ...
        class LLMPctxSoloAgent(LLMPctxMixin, LLMSoloAgent): ...

    Concrete subclasses must call self._init_pctx(env) before super().__init__().
    """

    def _init_pctx(self, env: Environment, tools: list[Callable] = None) -> list:
        """Set up pctx state and return the tau_tools list to pass to super().__init__()."""
        self.env = env
        self.internal_messages: list[Message] = []
        self.current_execute_callbacks: list[tuple[ToolCall, ToolMessage]] = []
        self.pctx_mode = os.environ.get("PCTX_MODE", "code").lower()

        # Create a persistent event loop for this agent instance
        self._loop = asyncio.new_event_loop()

        # convert env tools to pctx tools for code-mode registration
        pctx_tools = [
            pctx_tool(self._track_pctx_tool(fn), namespace=self.env.domain_name)
            for fn in tools
        ]
        self.pctx = Pctx(tools=pctx_tools)
        self.code_mode_fns = self._get_sync_code_mode_fns()

        tau_tools = [as_tool(t) for t in self.code_mode_fns.values()]
        for t in tools:
            tool = as_tool(t)
            tool.name = f"{self.env.domain_name}.{tool.name}"
            tau_tools.append(tool)

        return tau_tools

    def __del__(self):
        """Clean up the persistent event loop when the agent is deleted."""
        if hasattr(self, "_loop") and not self._loop.is_closed():
            self._loop.close()

    @property
    def system_prompt(self) -> str:
        """Append fs mode context to the parent's system prompt when applicable."""
        base_prompt = super().system_prompt

        if self.pctx_mode == "fs":
            function_list = "\n".join([f"- {fn.__name__}" for fn in self.tools])
            fs_context = PCTX_FS_ADDENDUM.format(
                namespace=self.env.domain_name, function_list=function_list
            )
            return base_prompt + "\n\n" + fs_context

        return base_prompt

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
            tool_msg = self.env.get_response(tool_call)

            tool_content = tool_msg.content
            if tool_content is not None:
                try:
                    tool_content = json.loads(tool_content)
                except json.decoder.JSONDecodeError:
                    pass

            logger.debug(
                f"[PCTX] Env Response - {tool_call.name} (error={tool_msg.error})\n{tool_msg.content}"
            )

            self.current_execute_callbacks.append((tool_call, tool_msg))

            return tool_content

        return tracked

    def _handle_pctx_tool_call(self, tool_call: ToolCall) -> ToolMessage:
        error = False
        logger.debug(f"[PCTX] Call - {tool_call.name}")
        try:
            resp = self.code_mode_fns[tool_call.name](**tool_call.arguments)
        except Exception as e:
            resp = f"Error: {e}"
            error = True

        logger.debug(f"[PCTX] Response - {tool_call.name} (error={error})")

        return ToolMessage(
            id=tool_call.id,
            content=json.dumps(resp),
            requestor=tool_call.requestor,
            role="tool",
            error=error,
        )

    def _run_in_loop(self, coroutine: Coroutine):
        """Run a coroutine in the agent's persistent event loop from sync code."""
        asyncio.set_event_loop(self._loop)
        try:
            return self._loop.run_until_complete(coroutine)
        finally:
            asyncio.set_event_loop(None)

    def _get_sync_code_mode_fns(self) -> dict[str, Callable]:
        """Get synchronous wrapper functions for pctx tools.

        Mode is controlled by PCTX_MODE environment variable:
        - "code" (default): Traditional discovery workflow (list_functions, get_function_details, execute)
        - "fs": Filesystem exploration workflow (execute_bash, execute_typescript)
        """

        def pctx_execute_typescript(code: str) -> str:
            return self._run_in_loop(self.pctx.execute(code)).markdown()

        pctx_execute_typescript.__doc__ = """Execute TypeScript code with access to all available tools.

CODE STRUCTURE:
async function run() {
    // Your code here
    // Call `await invoke({ name: "tool_name", arguments: {...} })` with proper types
    return result;
}

IMPORTANT RULES:
- ALWAYS make all tool calls via typescript unless otherwise explicitly stated in the tool definition.
- `invoke` takes a single object as an argument with 2 properties:
    - `name: string` - the name of the function/tool to be called, exactly as it is written in the function list.
    - `arguments: {[key: string]: any}` - the arguments for the function as described by the json schema in the function/tool list
- `invoke` will either return a native typescript object as defined by the successful return schema (in the function definition).
    - if there is no return schema in the function definition then the return type of the function is `unknown`.
- You can call any of the available tools using the `invoke` typescript function (does not need to be imported)
- You MUST define a `run()` function
- You MUST NOT call or export any functions from the root of the script, `run()` will be called automatically
- ALWAYS batch multiple tool calls into ONE execute typescript call
- Only listed tools are available to call via `invoke` - other common functions/modules like fetch(), fs, or other Node/Deno APIs are unavailable.
- Variables don't persist between executions
- Code runs in an isolated Deno sandbox

TOKEN USAGE WARNING:
- This tool could return LARGE responses if your code returns big objects
- Filter/map/reduce data IN YOUR CODE before returning
- Only return specific fields you need
- Use console.log() for intermediate results

RETURN TYPE NOTE:
- Function results are already parsed JavaScript objects, NOT JSON strings
- Do NOT call JSON.parse() on results
- Access properties directly (e.g., result.data)"""

        return {
            "pctx_execute_typescript": pctx_execute_typescript,
        }

        # mode = os.environ.get("PCTX_MODE", "code").lower()

        # if mode == "fs":
        #     # Filesystem mode: bash exploration + typescript execution
        #     def pctx_execute_bash(command: str) -> str:
        #         return self._run_in_loop(self.pctx.execute_bash(command)).markdown()

        #     pctx_execute_bash.__doc__ = PRESCRIPTIVE_DESCRIPTIONS["execute_bash"]

        #     def pctx_execute_typescript(code: str) -> str:
        #         return self._run_in_loop(self.pctx.execute(code)).markdown()

        #     pctx_execute_typescript.__doc__ = PRESCRIPTIVE_DESCRIPTIONS[
        #         "execute_typescript"
        #     ]

        #     return {
        #         "pctx_execute_bash": pctx_execute_bash,
        #         "pctx_execute_typescript": pctx_execute_typescript,
        #     }
        # else:
        #     # Code mode (default): Traditional discovery workflow
        #     def pctx_list_functions() -> str:
        #         return self._run_in_loop(self.pctx.list_functions()).code

        #     pctx_list_functions.__doc__ = PRESCRIPTIVE_DESCRIPTIONS["list_functions"]

        #     def pctx_get_function_details(functions: list[str]) -> str:
        #         return self._run_in_loop(self.pctx.get_function_details(functions)).code

        #     pctx_get_function_details.__doc__ = PRESCRIPTIVE_DESCRIPTIONS[
        #         "get_function_details"
        #     ]

        #     def pctx_execute(code: str) -> str:
        #         return self._run_in_loop(self.pctx.execute(code)).markdown()

        #     pctx_execute.__doc__ = PRESCRIPTIVE_DESCRIPTIONS["execute"]

        #     return {
        #         "pctx_list_functions": pctx_list_functions,
        #         "pctx_get_function_details": pctx_get_function_details,
        #         "pctx_execute": pctx_execute,
        #     }

    def connect(self):
        try:
            self._run_in_loop(self.pctx.connect())
            logger.debug(
                f"[PCTX] - connected to server: session_id={self.pctx._session_id}"
            )
        except Exception as e:
            logger.error(f"[PCTX] - connect failed: {e}")
            raise  # Re-raise to fail the task if we can't connect

    def disconnect(self):
        try:
            self._run_in_loop(self.pctx.disconnect())
            logger.debug(f"[PCTX] - disconnected from server")
        except Exception as e:
            logger.warning(f"[PCTX] - disconnect failed (ignoring): {e}")
            # Ignore disconnect errors - the session work is already done

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: LLMAgentState
    ) -> tuple[AssistantMessage, LLMAgentState]:
        msg, state = super().generate_next_message(message=message, state=state)

        # TODO: track these internal messages
        iteration = 0
        while msg.is_tool_call():
            iteration += 1
            msg_content = "\n\tmessage content: " + msg.content if msg.content else ""
            logger.debug(
                f"[PCTX] Internal messaging turn {iteration}\n\ttool call(s): {len(msg.tool_calls)}{msg_content}"
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


class LLMPctxAgent(LLMPctxMixin, LLMAgent):
    def __init__(
        self,
        env: Environment,
        task: Task,
        tools: list[Callable],
        llm: str | None = None,
        llm_args: dict | None = None,
    ):
        tau_tools = self._init_pctx(env, tools)
        super().__init__(tau_tools, env.get_policy(), llm, llm_args)


class LLMPctxSoloAgent(LLMPctxMixin, LLMSoloAgent):
    def __init__(
        self,
        env: Environment,
        task: Task,
        tools: list[Callable],
        llm: str | None = None,
        llm_args: dict | None = None,
    ):
        tau_tools = self._init_pctx(env, tools)
        print(tau_tools)
        super().__init__(tau_tools, env.get_policy(), task, llm, llm_args)
