"""Workspace file tools shared by all agents.

Reads are allowed anywhere inside the project (so agents can read `examples/`);
writes are confined to the workspace directory.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ..config import PROJECT_ROOT, workspace_dir

MAX_READ_CHARS = 60_000


def resolve_read_path(path: str) -> Path:
    """Resolve a user/agent supplied path for reading.

    Relative paths are tried against the workspace first, then the project root.
    """
    p = Path(path)
    candidates = [p] if p.is_absolute() else [workspace_dir() / p, PROJECT_ROOT / p]
    for c in candidates:
        c = c.resolve()
        if c.exists():
            if not (_is_within(c, PROJECT_ROOT) or _is_within(c, workspace_dir())):
                raise PermissionError(f"{path} is outside the project and workspace")
            return c
    raise FileNotFoundError(f"{path} not found in workspace or project")


def resolve_write_path(path: str) -> Path:
    ws = workspace_dir()
    p = Path(path)
    target = (p if p.is_absolute() else ws / p).resolve()
    if not _is_within(target, ws):
        raise PermissionError(f"writes are only allowed inside the workspace ({ws})")
    return target


def _is_within(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _rel(p: Path) -> str:
    for root in (workspace_dir(), PROJECT_ROOT):
        if _is_within(p, root):
            return p.relative_to(root.resolve()).as_posix()
    return str(p)


# ---------------------------------------------------------------- ADK tools


def list_files(subdir: str = "") -> dict:
    """List files in the workspace (or a subdirectory of it).

    Args:
        subdir: Optional subdirectory of the workspace. Use "" for the root.

    Returns:
        dict with `workspace` (absolute path) and `files` (relative paths).
    """
    ws = workspace_dir()
    base = (ws / subdir).resolve() if subdir else ws
    if not _is_within(base, ws) or not base.exists():
        return {"status": "error", "error": f"no such workspace directory: {subdir}"}
    files = sorted(
        p.relative_to(ws).as_posix()
        for p in base.rglob("*")
        if p.is_file() and "build" not in p.relative_to(ws).parts
    )
    return {"status": "ok", "workspace": str(ws), "files": files}


def read_file(path: str) -> dict:
    """Read a text file from the workspace or the project (e.g. examples/gemm.c).

    Args:
        path: Path relative to the workspace or project root, or absolute.

    Returns:
        dict with `content`, or `error`.
    """
    try:
        p = resolve_read_path(path)
        text = p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001 - report every failure to the model
        return {"status": "error", "error": str(e)}
    truncated = len(text) > MAX_READ_CHARS
    return {
        "status": "ok",
        "path": _rel(p),
        "content": text[:MAX_READ_CHARS],
        "truncated": truncated,
    }


def normalize_content(content: str) -> tuple[str, bool]:
    """Undo a common LLM tool-calling mistake: passing file contents as a
    JSON-encoded string (wrapped in quotes, with literal \\n escapes and no
    real newlines). Returns (content, was_normalized)."""
    s = content.strip()
    if "\n" in s or "\\n" not in s:
        return content, False
    if s.startswith('\\"') and s.endswith('\\"'):  # doubly-escaped outer quotes
        s = s[2:-2]
    decoded = None
    if s.startswith('"') and s.endswith('"'):
        try:
            decoded = json.loads(s)
        except json.JSONDecodeError:
            decoded = None
    if decoded is None:  # bare escaped text, or quoted text that isn't valid JSON
        body = s[1:-1] if s.startswith('"') and s.endswith('"') else s
        decoded = re.sub(r'\\(["\\nt])', lambda m: {"n": "\n", "t": "\t"}.get(m.group(1), m.group(1)), body)
    if isinstance(decoded, str) and "\n" in decoded:
        return decoded.rstrip("\n") + "\n", True
    return content, False


def write_file(path: str, content: str) -> dict:
    """Write a text file into the workspace, creating folders as needed.

    Args:
        path: Path relative to the workspace, e.g. "gemm/gemm_hls.cpp".
        content: Full file contents.

    Returns:
        dict with the written `path` and `bytes`.
    """
    content, normalized = normalize_content(content)
    try:
        p = resolve_write_path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}
    out = {"status": "ok", "path": _rel(p), "bytes": len(content.encode("utf-8")), "lines": content.count("\n")}
    if normalized:
        out["note"] = "Content arrived as an escaped string (literal \\n); it was decoded into real lines. Pass plain source text next time."
    return out
