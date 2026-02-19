uv run tau2 run \
    --domain telecom \
    --agent llm_agent_pctx_solo \
    --agent-llm "openrouter/openai/gpt-5" \
    --user "dummy_user" \
    --log-level DEBUG \
    --task-ids "[mobile_data_issue]data_mode_off|data_usage_exceeded[PERSONA:None]"
    # 0 1 2 3 4 5 6 7 8 9 10