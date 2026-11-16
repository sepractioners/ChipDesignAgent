"""Run the multi-agent flow from the command line (no web UI).

    python run_flow.py examples/gemm.c
    python run_flow.py examples/jacobi2d.c --prompt "Only profile and suggest optimizations."

Prints every agent hand-off and tool call as it happens, then the final report.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "agents"))


def _load_env() -> None:
    import os

    env = ROOT / "agents" / "chip_design_agent" / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def _short(obj, n=300) -> str:
    s = obj if isinstance(obj, str) else json.dumps(obj, default=str)
    return s if len(s) <= n else s[:n] + " ..."


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("kernel", help="C/C++ kernel to accelerate (copied into the workspace)")
    ap.add_argument("--prompt", default="", help="extra instructions for the orchestrator")
    args = ap.parse_args()

    _load_env()
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    from google.genai import types

    from chip_design_agent.agent import root_agent
    from chip_design_agent.config import workspace_dir

    src = Path(args.kernel).resolve()
    dst = workspace_dir() / src.name
    shutil.copy(src, dst)
    request = (
        f"Accelerate the kernel in '{src.name}' (workspace file). "
        "Profile it, optimize it, write an HLS version and verify it against the original. "
        + args.prompt
    )

    app = "chip_design_agent"
    sessions = InMemorySessionService()
    session = await sessions.create_session(app_name=app, user_id="cli")
    runner = Runner(agent=root_agent, app_name=app, session_service=sessions)
    msg = types.Content(role="user", parts=[types.Part(text=request)])

    print(f"> {request}\n")
    try:
        async for event in runner.run_async(user_id="cli", session_id=session.id, new_message=msg):
            for call in event.get_function_calls() or []:
                print(f"[{event.author}] -> {call.name}({_short(call.args)})")
            for resp in event.get_function_responses() or []:
                print(f"[{event.author}] <- {resp.name}: {_short(resp.response)}")
            if event.is_final_response() and event.content and event.content.parts:
                text = "".join(p.text or "" for p in event.content.parts)
                if text.strip():
                    print(f"\n=== {event.author} ===\n{text}\n")
    except Exception as e:  # noqa: BLE001 - report model/runtime failures without a traceback wall
        print(f"\n!!! Run stopped: {type(e).__name__}: {str(e)[:300]}")
        print("    Raise CHIP_AGENT_TIMEOUT or use a GPU / smaller OLLAMA_NUM_CTX if this was a timeout.")
        print(f"Workspace: {workspace_dir()}")
        sys.exit(1)
    print(f"Workspace: {workspace_dir()}")


if __name__ == "__main__":
    asyncio.run(main())
