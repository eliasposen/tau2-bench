uv run tau2 run \
    --domain airline \
    --agent llm_agent_pctx \
    --agent-llm "openrouter/google/gemini-3-flash-preview" \
    --user-llm "openrouter/openai/gpt-oss-120b" \
    --task-ids 1 \
    --log-level DEBUG