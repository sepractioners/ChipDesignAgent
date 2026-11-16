"""Optimization tools.

Includes a built-in directive suggester that always works, plus wrappers for
external design-space-exploration tools. The external tools are only used when
they are configured through environment variables; otherwise the wrappers
return `not_available` with setup instructions, so the agent can carry on.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path, PureWindowsPath

from ..config import workspace_dir
from .files import resolve_read_path
from .profiling import analyze_source

OPTIMIZERS = {
    "autosa": {
        "what": "Polyhedral systolic-array compiler (UCLA VAST). C with #pragma scop -> HLS C systolic array.",
        "best_for": ["gemm", "convolution", "other uniform-dependence loop nests"],
        "env": ["AUTOSA_ROOT", "AUTOSA_CMD_PREFIX (e.g. 'wsl' on Windows)"],
        "tool": "run_autosa",
    },
    "harp": {
        "what": "GNN-based surrogate model that predicts HLS QoR to guide pragma design-space exploration.",
        "best_for": ["pragma tuning for any kernel"],
        "env": ["HARP_CMD (template with {kernel} and {output})"],
        "tool": "run_external_optimizer",
    },
    "prometheus": {
        "what": "Holistic dataflow/tiling/pragma optimizer for multi-kernel FPGA designs (polyhedral + NLP solver).",
        "best_for": ["gemm", "stencil", "kernel chains (dataflow)"],
        "env": ["PROMETHEUS_CMD (template with {kernel} and {output})"],
        "tool": "run_external_optimizer",
    },
    "lift": {
        "what": "Functional IR with rewrite rules that explores algorithmic variants and lowers to code.",
        "best_for": ["stencil", "linear algebra expressed functionally"],
        "env": ["LIFT_CMD (template with {kernel} and {output})"],
        "tool": "run_external_optimizer",
    },
}


def _available(name: str) -> bool:
    if name == "autosa":
        return bool(os.getenv("AUTOSA_ROOT"))
    return bool(os.getenv(f"{name.upper()}_CMD"))


def _largest_divisor(n: int | None, cap: int) -> int:
    if not n:
        return cap
    for d in range(min(n, cap), 0, -1):
        if n % d == 0:
            return d
    return 1


def suggest_for_nest(nest: dict) -> dict:
    pat = nest["pattern"]
    kind = pat["pattern"]
    loops = nest["loops"]
    inner = loops[-1]
    s: dict = {"function": nest["function"], "loop_line": loops[0]["line"], "pattern": kind, "directives": [], "restructuring": []}

    if kind == "gemm":
        d = pat["dims"]
        ti, tj, tk = _largest_divisor(d["M"], 32), _largest_divisor(d["N"], 32), _largest_divisor(d["K"], 32)
        lat = _largest_divisor(min(ti, tj), 8)
        simd = _largest_divisor(tk, 4)
        s["restructuring"] = [
            f"Tile i/j/k by {ti}/{tj}/{tk}; keep a local tile of {pat['output']} in on-chip memory.",
            f"Reorder to i-k-j (reduction var '{pat['reduction_var']}' outside the innermost loop) so the inner loop has no loop-carried dependence.",
            "Burst-copy tiles of the inputs into local buffers before computing.",
        ]
        s["directives"] = [
            "#pragma HLS PIPELINE II=1   // on the innermost j loop",
            f"#pragma HLS ARRAY_PARTITION variable=<local {pat['output']} tile> type=cyclic factor={min(tj, 16)} dim=2",
            f"#pragma HLS UNROLL factor={min(tj, 16)}  // optional, matches the partition factor",
        ]
        s["autosa_sa_sizes"] = (
            f"{{kernel[]->space_time[3];kernel[]->array_part[{ti},{tj},{tk}];"
            f"kernel[]->latency[{lat},{lat}];kernel[]->simd[{simd}]}}"
        )
        s["expected"] = "Throughput bound by #PEs x SIMD; a 2-D systolic array via AutoSA usually beats hand pragmas."
    elif kind == "stencil":
        r = pat["radius"]
        s["restructuring"] = [
            f"Stream {pat['input']} row by row and keep {2 * r + 1} rows in line buffers.",
            f"Use a {2 * r + 1}x{2 * r + 1} register window shifted every cycle so each input element is read from DRAM once.",
        ]
        if pat.get("in_place"):
            s["restructuring"].append("The update is in place (Gauss-Seidel style); double-buffer or keep the sequential order, or results will differ.")
        s["directives"] = [
            f"#pragma HLS PIPELINE II=1   // on the innermost '{inner['var']}' loop",
            "#pragma HLS ARRAY_PARTITION variable=window type=complete",
            f"#pragma HLS ARRAY_PARTITION variable=line_buf type=complete dim=1   // {2 * r} rows",
            "#pragma HLS DATAFLOW   // if load / compute / store are split into functions",
        ]
        s["expected"] = "One output per cycle after the pipeline fills; DRAM traffic drops to ~1 read + 1 write per point."
    elif kind == "reduction":
        s["restructuring"] = ["Accumulate into 8 independent partial sums, then add them at the end (breaks the floating-point add dependence)."]
        s["directives"] = ["#pragma HLS PIPELINE II=1", "#pragma HLS ARRAY_PARTITION variable=partial type=complete"]
        s["expected"] = "II drops from the FP-adder latency (~4-7) to 1."
    elif kind == "elementwise":
        u = _largest_divisor(inner["trip_count"], 8)
        s["directives"] = [
            "#pragma HLS PIPELINE II=1",
            f"#pragma HLS UNROLL factor={u}  // partition the arrays cyclically by the same factor",
        ]
        s["expected"] = f"~{u} results per cycle if memory bandwidth allows."
    else:
        s["directives"] = ["#pragma HLS PIPELINE II=1   // on the innermost loop, then inspect the II violation report"]
        s["expected"] = "Unknown pattern: start with pipelining and iterate using synthesis reports."
    return s


# ---------------------------------------------------------------- ADK tools


def list_optimizers() -> dict:
    """List the optimization back-ends (AutoSA, HARP, Prometheus, LIFT), what each
    is good for, and whether it is configured on this machine.

    Returns:
        dict with `optimizers` (name -> what/best_for/available/tool) and
        `builtin` describing the always-available `suggest_directives` tool.
    """
    return {
        "status": "ok",
        "optimizers": {k: {**v, "available": _available(k)} for k, v in OPTIMIZERS.items()},
        "builtin": "suggest_directives: pattern-based tiling/pipelining/partitioning advice (always available)",
        "synthesis_available": _vitis_hls() is not None,
    }


def suggest_directives(path: str) -> dict:
    """Suggest HLS restructuring and pragmas for every loop nest in a kernel,
    based on its matched pattern (gemm -> tiling/systolic, stencil -> line
    buffers, reduction -> partial sums, elementwise -> pipeline+unroll).
    Also returns AutoSA `sa_sizes` for gemm nests.

    Args:
        path: Kernel source file (relative to workspace or project root).

    Returns:
        dict with `suggestions`, one per loop nest, hottest first.
    """
    try:
        src = resolve_read_path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}
    nests = analyze_source(src)["loop_nests"]
    return {"status": "ok", "suggestions": [suggest_for_nest(n) for n in nests]}


def _to_wsl(p: Path) -> str:
    w = PureWindowsPath(p)
    if not w.drive:
        return p.as_posix()
    return f"/mnt/{w.drive[0].lower()}" + "/" + "/".join(w.parts[1:])


def run_autosa(kernel_path: str, sa_sizes: str, simd_info_path: str = "") -> dict:
    """Run AutoSA to generate a systolic-array HLS design for a kernel.
    The kernel's target loop nest must be wrapped in `#pragma scop` /
    `#pragma endscop`.

    Args:
        kernel_path: C kernel file (relative to workspace or project root).
        sa_sizes: AutoSA --sa-sizes string, e.g. from suggest_directives.
        simd_info_path: Optional AutoSA simd_info.json path ("" to skip).

    Returns:
        dict with `status` (ok / error / not_available), command output, and generated files.
    """
    root = os.getenv("AUTOSA_ROOT")
    if not root:
        return {
            "status": "not_available",
            "message": "AutoSA is not configured. Install it (Linux / WSL: https://github.com/UCLA-VAST/AutoSA), "
            "then set AUTOSA_ROOT (path inside WSL) and AUTOSA_CMD_PREFIX=wsl in the .env file.",
        }
    try:
        kernel = resolve_read_path(kernel_path)
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}
    if "#pragma scop" not in kernel.read_text(encoding="utf-8", errors="replace"):
        return {"status": "error", "error": "Kernel has no '#pragma scop' region; wrap the target loop nest in #pragma scop / #pragma endscop."}

    out_dir = workspace_dir() / "autosa" / kernel.stem
    (out_dir / "src").mkdir(parents=True, exist_ok=True)
    prefix = os.getenv("AUTOSA_CMD_PREFIX", "").strip()
    conv = _to_wsl if prefix.startswith("wsl") else (lambda p: p.as_posix())
    cmd = (
        f"cd '{root}' && ./autosa '{conv(kernel)}' --config=./autosa_config/autosa_config.json "
        f"--target=autosa_hls_c --output-dir='{conv(out_dir)}' --sa-sizes=\"{sa_sizes}\" --host-serialize"
    )
    if simd_info_path:
        cmd += f" --simd-info='{conv(resolve_read_path(simd_info_path))}'"
    argv = (prefix.split() if prefix else []) + ["bash", "-lc", cmd]
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=900)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"status": "error", "error": str(e), "command": cmd}
    files = sorted(p.relative_to(workspace_dir()).as_posix() for p in out_dir.rglob("*") if p.is_file())
    return {
        "status": "ok" if proc.returncode == 0 else "error",
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "generated_files": files,
        "command": cmd,
    }


def run_external_optimizer(name: str, kernel_path: str) -> dict:
    """Run an external optimizer (harp, prometheus or lift) through the command
    template configured in the environment variable <NAME>_CMD.

    Args:
        name: One of "harp", "prometheus", "lift".
        kernel_path: Kernel file (relative to workspace or project root).

    Returns:
        dict with `status` (ok / error / not_available), output and generated files.
    """
    name = name.lower().strip()
    if name not in OPTIMIZERS or name == "autosa":
        return {"status": "error", "error": f"unknown optimizer {name!r}; use harp, prometheus or lift (AutoSA has run_autosa)"}
    template = os.getenv(f"{name.upper()}_CMD")
    if not template:
        return {
            "status": "not_available",
            "message": f"{name} is not configured. Set {name.upper()}_CMD to a command template using "
            "{kernel} and {output}. Use suggest_directives meanwhile.",
        }
    try:
        kernel = resolve_read_path(kernel_path)
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}
    out_dir = workspace_dir() / name / kernel.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = template.format(kernel=str(kernel), output=str(out_dir))
    try:
        proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired as e:
        return {"status": "error", "error": str(e), "command": cmd}
    files = sorted(p.relative_to(workspace_dir()).as_posix() for p in out_dir.rglob("*") if p.is_file())
    return {
        "status": "ok" if proc.returncode == 0 else "error",
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "generated_files": files,
        "command": cmd,
    }


def _vitis_hls() -> str | None:
    exe = shutil.which("vitis_hls") or shutil.which("vitis-run")
    if exe:
        return exe
    root = os.getenv("XILINX_HLS")
    if root:
        for name in ("vitis_hls.bat", "vitis_hls"):
            p = Path(root) / "bin" / name
            if p.exists():
                return str(p)
    return None


def run_hls_synthesis(source_path: str, top_function: str, part: str = "xcu250-figd2104-2L-e", clock_period_ns: float = 3.33) -> dict:
    """Run Vitis HLS C-synthesis on an HLS source file and return latency,
    initiation interval and resource estimates. Requires Vitis HLS.

    Args:
        source_path: HLS C/C++ file (relative to workspace).
        top_function: Name of the top-level function.
        part: FPGA part number.
        clock_period_ns: Target clock period in ns.

    Returns:
        dict with `latency_cycles`, `interval`, `resources`, `available_resources`, or not_available.
    """
    exe = _vitis_hls()
    if not exe:
        return {"status": "not_available", "message": "Vitis HLS not found; add it to PATH or set XILINX_HLS."}
    try:
        src = resolve_read_path(source_path)
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}
    proj_dir = workspace_dir() / "hls_proj" / src.stem
    proj_dir.mkdir(parents=True, exist_ok=True)
    tcl = proj_dir / "run.tcl"
    tcl.write_text(
        f"open_project -reset proj\nset_top {top_function}\nadd_files {{{src.as_posix()}}}\n"
        f"open_solution -reset sol -flow_target vitis\nset_part {{{part}}}\ncreate_clock -period {clock_period_ns}\n"
        "csynth_design\nexit\n",
        encoding="utf-8",
    )
    try:
        proc = subprocess.run([exe, "-f", "run.tcl"], cwd=proj_dir, capture_output=True, text=True, timeout=3600)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"status": "error", "error": str(e)}
    report = proj_dir / "proj" / "sol" / "syn" / "report" / f"{top_function}_csynth.xml"
    if proc.returncode != 0 or not report.exists():
        errs = [l for l in proc.stdout.splitlines() if "ERROR" in l]
        return {"status": "error", "returncode": proc.returncode, "errors": errs[-20:], "log_tail": proc.stdout[-3000:]}
    root = ET.parse(report).getroot()

    def txt(path):
        el = root.find(path)
        return el.text.strip() if el is not None and el.text else None

    def res(path):
        node = root.find(path)
        return {c.tag: c.text for c in node} if node is not None else {}

    return {
        "status": "ok",
        "estimated_clock_ns": txt(".//PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod"),
        "latency_cycles": {
            "best": txt(".//SummaryOfOverallLatency/Best-caseLatency"),
            "worst": txt(".//SummaryOfOverallLatency/Worst-caseLatency"),
        },
        "interval": txt(".//SummaryOfOverallLatency/Interval-min"),
        "resources": res(".//AreaEstimates/Resources"),
        "available_resources": res(".//AreaEstimates/AvailableResources"),
        "warnings": [l for l in proc.stdout.splitlines() if re.search(r"WARNING: \[HLS 200-88[05]", l)][:10],
    }
