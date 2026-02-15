# Running tau2 Evaluation on Modal

This guide explains how to run your tau2 evaluation benchmark on Modal's serverless platform.

The Modal setup automatically handles:
- Installing and starting the pctx server (https://github.com/portofcontext/pctx)
- Installing all Python dependencies using `uv` (fast!)
- Running your evaluation with the configured parameters

## Prerequisites

### 1. Ensure modal is in dependencies

Modal should already be in `pyproject.toml`. If not, add it:

```bash
uv add modal
```

This will add the latest version of Modal to your dependencies.

### 2. Install Modal CLI

If you're using `uv`, Modal is already installed. Otherwise:

```bash
uv pip install modal
```

### 3. Authenticate with Modal

```bash
modal setup
```

This will open a browser window to create/login to your Modal account (free to start).

## Setup Secrets

Create a Modal secret for your OpenRouter API key:

```bash
modal secret create openrouter OPENROUTER_API_KEY=your-api-key-here
```

**Important:** Replace `your-api-key-here` with your actual OpenRouter API key.

To verify your secret was created:
```bash
modal secret list
```

## Running the Evaluation

To run your evaluation on Modal:

```bash
modal run modal_run.py
```

This will:
1. Build a container image with all dependencies
2. Copy your local code (tau2-bench and pctx-client) into the image
3. Upload the image to Modal's cloud
4. Start a remote container with the pctx server
5. Install Python dependencies using `uv` (fast!)
6. Run the tau2 evaluation with your configured parameters
7. Stream all output back to your terminal in real-time
8. Clean up the pctx server when done

**Note:** The first run will take longer (~5-10 minutes) as it builds the image. Subsequent runs will be much faster thanks to caching.

## Customizing the Evaluation

### Changing Evaluation Parameters

Edit `modal_run.py` and modify the arguments in the `run_tau2_eval()` function (around line 93):

```python
[
    "tau2", "run",
    "--domain", "airline",           # Change domain here
    "--agent", "llm_agent_pctx",     # Change agent type here
    "--agent-llm", "openrouter/openai/gpt-5",
    "--user-llm", "openrouter/openai/gpt-4o-2024-05-13",
    "--log-level", "INFO",
    "--task-ids", "7"                # Change task IDs here
]
```

### Changing Resources

To adjust CPU/memory, edit the `@app.function` decorator (around line 36):

```python
@app.function(
    image=image,
    secrets=[modal.Secret.from_name("openrouter")],
    timeout=3600 * 6,  # 6 hour timeout
    cpu=2.0,           # Change CPU cores here
    memory=4096,       # Change memory (MB) here
)
```

## Running in the Background

To deploy the app and run it detached from your terminal:

```bash
modal deploy modal_run.py
```

Then trigger runs from the Modal dashboard at https://modal.com/apps

## Viewing Logs

All output from your evaluation will be streamed to your terminal. You can also:
- View logs in the Modal web dashboard at https://modal.com
- Check the function logs for any errors or debugging info

## Cost

Modal charges based on compute time. The current configuration uses:
- **CPU:** 2 cores
- **Memory:** 4GB RAM
- **Max timeout:** 6 hours per run

Check current Modal pricing at https://modal.com/pricing

The free tier includes some credits to get started.

## Important Notes

### Local Dependencies

The script requires access to:
- `/Users/patrickkelly/repos/portofcontext/benchmarking/tau2-bench` (this repo)
- `/Users/patrickkelly/repos/portofcontext/pctx/pctx-py` (pctx-client)

**If you're running on a different machine**, update the pctx-py path in `modal_run.py` (line 30-33):

```python
.add_local_dir(
    "/path/to/your/pctx-py",  # Update this path!
    "/root/pctx-py"
)
```

### Environment Variables

The script sets:
- `PCTX_MODE=fs` - Automatically configured
- `OPENROUTER_API_KEY` - From your Modal secret

### Results Storage

Results are saved in the `/root/tau2-bench/data` directory on the Modal container. This is ephemeral by default. To persist results:
1. Add a Modal Volume to the function
2. Or download results before the container terminates
3. Or write results to an external service (S3, etc.)

## Troubleshooting

### "Secret not found" error

Check your secrets:
```bash
modal secret list
```

If `openrouter` is missing, create it:
```bash
modal secret create openrouter OPENROUTER_API_KEY=your-key-here
```

### "Module 'modal' has no attribute..." errors

Make sure you're using Modal >= 1.3.3:
```bash
uv add modal  # Updates to latest
```

### Build errors during image creation

Modal uses the current directory when building. Make sure:
1. You're in the tau2-bench directory
2. The pctx-py path is correct
3. Both directories exist and are readable

### pctx server not starting

The script installs pctx using the official installer. If you see errors:
1. Check the Modal logs for installation failures
2. The pctx binary should be at `~/.cargo/bin/pctx`
3. Increase the startup sleep time (line 69) if needed

### Evaluation hangs or times out

- Default timeout is 6 hours
- Adjust the `timeout` parameter in `@app.function` if needed
- Check Modal logs for any errors during execution

## Getting Help

- Modal docs: https://modal.com/docs
- Modal Discord: https://discord.gg/modal
- pctx repo: https://github.com/portofcontext/pctx
