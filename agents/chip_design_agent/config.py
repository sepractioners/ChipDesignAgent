"""Model and path configuration.

Everything is driven by environment variables (loaded from `.env` by ADK, or
by the CLI) so the same agents can run on local Ollama or a hosted model.
"""

from __future__ import annotations

import os
from pathlib import Path

from google.adk.models.lite_llm import LiteLlm

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent.parent

DEFAULT_MODEL = "ollama_chat/gemma4"

# LiteLLM reads this to find the Ollama server.
os.environ.setdefault("OLLAMA_API_BASE", "http://localhost:11434")


def workspace_dir() -> Path:
    """Directory the agents may write into (kernels, HLS code, testbenches)."""
    ws = Path(os.getenv("CHIP_AGENT_WORKSPACE", PROJECT_ROOT / "workspace")).resolve()
    ws.mkdir(parents=True, exist_ok=True)
    return ws


def get_model(role: str = ""):
    """Return the model for an agent.

    `CHIP_AGENT_MODEL_<ROLE>` (e.g. CHIP_AGENT_MODEL_TRANSFORM) overrides the
    global `CHIP_AGENT_MODEL` for a single agent, so you can run a bigger model
    only where it matters.

    - `gemini-*` names are passed to ADK natively.
    - Anything else goes through LiteLLM, e.g. `ollama_chat/gemma4`,
      `anthropic/claude-sonnet-5-5`, `openai/...`.

    Use the `ollama_chat/` prefix (not `ollama/`) for Ollama: it is the one
    that supports tool calling correctly through LiteLLM.
    """
    name = None
    if role:
        name = os.getenv(f"CHIP_AGENT_MODEL_{role.upper()}")
    name = name or os.getenv("CHIP_AGENT_MODEL", DEFAULT_MODEL)
    if name.startswith("gemini"):
        return name
    kwargs: dict = {"max_tokens": int(os.getenv("CHIP_AGENT_MAX_TOKENS", "8192"))}
    local = name.startswith(("ollama_chat/", "ollama/"))
    if local:
        # Ollama's default context window is small; a code-writing agent sees
        # its instructions, the source file, tool results and its own output.
        kwargs["num_ctx"] = int(os.getenv("OLLAMA_NUM_CTX", "32768"))
    # Seconds per model call. CPU-only local inference can exceed LiteLLM's 600 s default.
    kwargs["timeout"] = float(os.getenv("CHIP_AGENT_TIMEOUT", "1800" if local else "600"))
    return LiteLlm(model=name, **kwargs)
