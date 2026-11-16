"""Code-transformation helpers: HLS synthesizability lint + pragma reference."""

from __future__ import annotations

import re

from . import cparse
from .files import resolve_read_path

PRAGMA_REFERENCE = """\
Vitis HLS pragma cheat sheet (put inside the function / loop body they apply to):

Interfaces (top function only)
  #pragma HLS INTERFACE mode=m_axi port=A bundle=gmem0 depth=<elements>
  #pragma HLS INTERFACE mode=s_axilite port=n
  #pragma HLS INTERFACE mode=s_axilite port=return

Loops
  #pragma HLS PIPELINE II=1            // start one iteration per cycle
  #pragma HLS UNROLL factor=4          // replicate the body; omit factor = full unroll
  #pragma HLS LOOP_TRIPCOUNT min=64 max=64   // only for reports when bounds are variable
  #pragma HLS LOOP_FLATTEN off

Arrays / memories
  #pragma HLS ARRAY_PARTITION variable=buf type=cyclic factor=4 dim=2
  #pragma HLS ARRAY_PARTITION variable=buf type=complete dim=1
  #pragma HLS BIND_STORAGE variable=buf type=ram_2p impl=bram

Task-level
  #pragma HLS DATAFLOW                 // overlap load -> compute -> store functions
  hls::stream<T> s; #pragma HLS STREAM variable=s depth=16

Rules of thumb
  * Copy off-chip (m_axi) arrays into local buffers in bursts, compute on local buffers, write back.
  * Unrolling by U needs the accessed array dimension partitioned by >= U (cyclic) or completely.
  * A loop-carried dependence on an accumulator limits II: use partial sums
    (e.g. sum[8], then a final reduction) for floating point.
  * GEMM: tile i/j, keep a local C tile, move the k loop outward, pipeline the j loop.
  * Stencil: use line buffers + a sliding window so each input is read from DRAM once.
  * Keep loop bounds compile-time constants; no malloc/new, recursion, or STL containers.
  * Guard host-only code with #ifndef __SYNTHESIS__.
"""

_RULES = [
    (r"\b(malloc|calloc|realloc|free)\s*\(", "error", "Dynamic memory allocation is not synthesizable; use fixed-size arrays."),
    (r"\bnew\b\s+\w|\bdelete\b\s*[\[\w]", "error", "new/delete is not synthesizable; use fixed-size arrays."),
    (r"std::(vector|map|unordered_map|list|set|string|deque)\b", "error", "STL containers are not synthesizable; use arrays or hls::stream."),
    (r"\b(fopen|fread|fwrite|fprintf|scanf|system|exit)\s*\(", "error", "System/file I/O is not synthesizable."),
    (r"\b(printf|puts)\s*\(|std::cout|std::cerr", "warning", "Console output is ignored by synthesis; wrap in #ifndef __SYNTHESIS__."),
    (r"\bgoto\b", "warning", "goto makes control flow hard to pipeline."),
    (r"\bthrow\b|\btry\s*\{", "error", "Exceptions are not synthesizable."),
    (r"\bvirtual\b", "error", "Virtual functions are not synthesizable."),
    (r"\bwhile\s*\(", "warning", "while loops have unknown trip count; prefer counted for-loops (or add LOOP_TRIPCOUNT)."),
    (r"\bdouble\b", "info", "double uses ~2-4x the DSPs of float; use float or ap_fixed if precision allows."),
]


def lint_source(src: str, top_function: str = "") -> dict:
    clean = cparse.strip_comments(src)
    issues = []
    lines = clean.splitlines()
    in_host_block = False
    for ln, line in enumerate(lines, 1):
        s = line.strip()
        if re.match(r"#\s*ifndef\s+__SYNTHESIS__", s):
            in_host_block = True
        elif in_host_block and re.match(r"#\s*(endif|else)", s):
            in_host_block = False
        if in_host_block or s.startswith("#"):
            continue
        for pat, sev, msg in _RULES:
            if re.search(pat, line):
                issues.append({"line": ln, "severity": sev, "message": msg, "code": s[:120]})

    # Loop pragmas must be the first statement INSIDE the loop body they target.
    code_lines = [(ln, l.strip()) for ln, l in enumerate(lines, 1) if l.strip()]
    for idx, (ln, s) in enumerate(code_lines[:-1]):
        m = re.match(r"#\s*pragma\s+HLS\s+(PIPELINE|UNROLL|LOOP_FLATTEN|LOOP_TRIPCOUNT)\b", s, flags=re.I)
        if not m:
            continue
        nxt = code_lines[idx + 1][1]
        if re.match(r"for\s*\(", nxt) or re.match(r"\w+\s*:\s*for\s*\(", nxt):
            issues.append({"line": ln, "severity": "warning", "message": f"{m.group(1).upper()} is placed just before a for-loop, so it applies to the ENCLOSING loop or function, not this loop. Move it to the first line inside this loop's body.", "code": s})

    funcs = cparse.parse_functions(src)
    names = {f.name for f in funcs}
    for f in funcs:
        if re.search(rf"\b{re.escape(f.name)}\s*\(", f.body):
            issues.append({"line": 0, "severity": "error", "message": f"Function {f.name} is recursive; recursion is not synthesizable.", "code": ""})
        for p in f.params:
            if p.pointer and not p.dims and f.name == top_function:
                has_iface = re.search(rf"INTERFACE[^\n]*port\s*=\s*{re.escape(p.name)}\b", f.body)
                if not has_iface:
                    issues.append({"line": 0, "severity": "warning", "message": f"Pointer argument '{p.name}' of top function has no INTERFACE pragma (add m_axi with depth=).", "code": ""})

    defines = cparse.parse_defines(clean)
    if not funcs:
        issues.append({"line": 0, "severity": "error", "message": "No C/C++ function definitions found. The file may have been written as an escaped string (literal \\n) or be truncated.", "code": ""})

    for f in funcs:
        issues.extend(_interface_issues(f, defines))
        has_loops = bool(cparse.parse_loops(f.body, defines))
        for m in re.finditer(r"#\s*pragma\s+HLS\s+PIPELINE\b(?![^\n]*\boff\b)", f.body, flags=re.I):
            before = f.body[: m.start()]
            if has_loops and before.count("{") - before.count("}") == 0:
                issues.append({"line": 0, "severity": "warning", "message": f"PIPELINE at the top level of '{f.name}' fully unrolls every loop inside it (huge hardware). Put PIPELINE inside the innermost loop instead.", "code": m.group(0)})

    for f in funcs:
        for top in cparse.parse_loops(f.body, defines):
            stack = [top]
            while stack:
                l = stack.pop()
                stack.extend(l.children)
                if l.var != "?" and l.trip_count is None and "LOOP_TRIPCOUNT" not in l.body:
                    issues.append({"line": l.line, "severity": "info", "message": f"Loop over '{l.var}' has a non-constant bound ({l.upper}); add LOOP_TRIPCOUNT or make it constant.", "code": ""})

    if top_function and top_function not in names:
        issues.append({"line": 0, "severity": "error", "message": f"Top function '{top_function}' not found.", "code": ""})
    pragmas = re.findall(r"#\s*pragma\s+HLS\s+(\w+)", src, flags=re.I)
    errors = sum(1 for i in issues if i["severity"] == "error")
    return {
        "synthesizable": errors == 0,
        "errors": errors,
        "warnings": sum(1 for i in issues if i["severity"] == "warning"),
        "issues": issues,
        "pragmas_used": dict((p.upper(), pragmas.count(p)) for p in set(pragmas)),
        "functions": sorted(names),
    }


def _interface_issues(f: cparse.Function, defines: dict) -> list[dict]:
    """Checks on m_axi ports: depth vs array size, and partitioning of off-chip arrays."""
    out = []
    params = {p.name: p for p in f.params}
    maxi = {}
    for m in re.finditer(r"#\s*pragma\s+HLS\s+INTERFACE\b([^\n]*)", f.body, flags=re.I):
        args = m.group(1)
        if not re.search(r"\bm_axi\b", args):
            continue
        port = re.search(r"\bport\s*=\s*(\w+)", args)
        if not port:
            continue
        depth = re.search(r"\bdepth\s*=\s*(\d+)", args)
        maxi[port.group(1)] = int(depth.group(1)) if depth else None
    for name, depth in maxi.items():
        p = params.get(name)
        if not p or not p.dims or depth is None:
            continue
        size = 1
        for d in p.dims:
            v = cparse.eval_expr(d, defines)
            if v is None:
                size = None
                break
            size *= int(v)
        if size and depth < size:
            out.append({"line": 0, "severity": "warning", "message": f"m_axi port '{name}' has depth={depth} but the array has {size} elements; co-simulation will access out of bounds. Use depth={size}.", "code": ""})
    for m in re.finditer(r"#\s*pragma\s+HLS\s+ARRAY_PARTITION\b[^\n]*\bvariable\s*=\s*(\w+)", f.body, flags=re.I):
        if m.group(1) in maxi:
            out.append({"line": 0, "severity": "error", "message": f"'{m.group(1)}' is an m_axi (off-chip) port and cannot be partitioned. Copy it into a local array and partition that.", "code": m.group(0).strip()})
    return out


# ---------------------------------------------------------------- ADK tools


def lint_hls(path: str, top_function: str = "") -> dict:
    """Check a C/C++ file for constructs that HLS tools cannot synthesize
    (malloc/new, recursion, STL containers, file I/O, exceptions...) and for
    missing interface pragmas / non-constant loop bounds.

    Args:
        path: Source file, relative to the workspace or project root.
        top_function: Name of the HLS top-level function ("" if unknown).

    Returns:
        dict with `synthesizable` (bool), `issues` (line, severity, message) and `pragmas_used`.
    """
    try:
        src = resolve_read_path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}
    out = lint_source(src, top_function)
    out["status"] = "ok"
    return out


def hls_pragma_reference() -> dict:
    """Return a concise Vitis HLS pragma cheat sheet and optimization rules of thumb.

    Returns:
        dict with `reference` text.
    """
    return {"status": "ok", "reference": PRAGMA_REFERENCE}
