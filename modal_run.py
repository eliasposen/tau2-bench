"""
Modal app to run tau2 benchmarks remotely.

Setup:
1. Install Modal: pip install modal
2. Authenticate: modal setup
3. Set secrets: modal secret create openrouter OPENROUTER_API_KEY=your-key-here
4. Run: modal run modal_run.py
"""

import modal

# Create Modal app
app = modal.App("tau2-pctx-eval")

# Create a custom image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.10")
    # Install curl and other system dependencies
    .apt_install("curl")
    # Install uv
    .pip_install("uv")
    # Install pctx server
    .run_commands(
        "curl --proto '=https' --tlsv1.2 -LsSf https://github.com/portofcontext/pctx/releases/download/v0.6.0-beta.1/pctx-installer.sh | sh"
    )
    # Add tau2-bench directory (exclude .git to avoid build errors)
    .add_local_dir(".", "/root/tau2-bench", ignore=[".git"])
)


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("openrouter")],
    timeout=3600 * 6,  # 6 hour timeout
    cpu=2.0,
    memory=4096,
)
def run_tau2_eval():
    """Run the tau2 evaluation benchmark."""
    import os
    import sys
    import subprocess
    import time

    # Set environment variables
    os.environ["PCTX_MODE"] = "fs"

    # Change to tau2-bench directory
    os.chdir("/root/tau2-bench")

    # Start pctx server in the background
    print("Starting pctx server...")
    # Add pctx to PATH (installed in ~/.local/bin by default)
    pctx_path = os.path.expanduser("~/.local/bin/pctx")

    # Start pctx server as a background process
    pctx_process = subprocess.Popen(
        [pctx_path, "start"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    # Give the server a moment to start up
    time.sleep(5)
    print("pctx server started")

    # Create a fresh virtual environment
    print("Creating virtual environment...")
    subprocess.run(["uv", "venv"], check=True, capture_output=False)

    # Install tau2 package and its dependencies (including pctx-client from PyPI)
    print("Installing tau2...")
    subprocess.run(
        ["uv", "pip", "install", "-e", "."], check=True, capture_output=False
    )

    # Run tau2 command using uv run (to use the venv we created)
    print("Running tau2 evaluation...")
    try:
        result = subprocess.run(
            [
                "uv",
                "run",
                "tau2",
                "run",
                "--domain",
                "airline",
                "--agent",
                "llm_agent_pctx",
                "--agent-llm",
                "openrouter/openai/gpt-5",
                "--user-llm",
                "openrouter/openai/gpt-4o-2024-05-13",
                "--log-level",
                "INFO",
                "--task-ids",
                "7",
            ],
            check=True,
            capture_output=False,
        )
        return_code = result.returncode
    finally:
        # Stop pctx server
        print("Stopping pctx server...")
        subprocess.run([pctx_path, "stop"], capture_output=True)
        pctx_process.terminate()
        pctx_process.wait()

    print("Evaluation complete!")
    return return_code


@app.local_entrypoint()
def main():
    """Entry point for modal run."""
    print("Starting tau2 evaluation on Modal...")
    result = run_tau2_eval.remote()
    print(f"Evaluation finished with return code: {result}")
