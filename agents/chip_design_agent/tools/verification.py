"""Testing / verification: testbench generation and C-simulation.

Equivalence check strategy: one translation unit #includes the reference
kernel and the HLS candidate (renaming the reference function with a macro so
both can share a name), fills every array argument with the same random data,
runs both, and compares the arrays the kernels write.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from ..config import workspace_dir
from . import cparse
from .files import resolve_read_path, resolve_write_path

_FLOAT_TYPES = {"float", "double", "half"}


def find_compiler() -> list[str] | None:
    """Return the argv prefix for a C++ compiler."""
    if os.getenv("CXX"):
        return os.getenv("CXX").split()
    for name in ("g++", "clang++"):
        if shutil.which(name):
            return [name]
    try:
        import ziglang  # noqa: F401

        return [sys.executable, "-m", "ziglang", "c++"]
    except ImportError:
        return None


def _parse_kv(text: str) -> dict[str, str]:
    out = {}
    for part in filter(None, (p.strip() for p in (text or "").split(","))):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def _written_params(fn: cparse.Function) -> set[str]:
    names = {p.name for p in fn.params}
    written = set()
    body = cparse.strip_comments(fn.body)
    for stmt in re.split(r"[;{}]", body):
        a = cparse.split_assignment(stmt)
        if not a:
            continue
        lhs = a[0]
        for acc in cparse.find_accesses(lhs):
            if acc.array in names:
                written.add(acc.array)
        m = re.match(r"^\*\s*\(?\s*(\w+)", lhs)  # *(p + i) = ...
        if m and m.group(1) in names:
            written.add(m.group(1))
    # passing an array to memcpy as destination
    for m in re.finditer(r"memcpy\s*\(\s*\(?\s*\w*\s*\*?\s*\)?\s*(\w+)", body):
        if m.group(1) in names:
            written.add(m.group(1))
    return written


def build_testbench(
    ref_src: str,
    ref_include: str,
    ref_func: str,
    cand_src: str,
    cand_include: str,
    cand_func: str,
    pointer_sizes: str = "",
    scalar_values: str = "",
    tolerance: float = 1e-4,
    seed: int = 42,
) -> tuple[str, list[str]]:
    ref_fns = {f.name: f for f in cparse.parse_functions(ref_src)}
    cand_fns = {f.name: f for f in cparse.parse_functions(cand_src)}
    if ref_func not in ref_fns:
        raise ValueError(f"reference function {ref_func!r} not found (found: {sorted(ref_fns)})")
    if cand_func not in cand_fns:
        raise ValueError(f"candidate function {cand_func!r} not found (found: {sorted(cand_fns)})")
    ref, cand = ref_fns[ref_func], cand_fns[cand_func]
    if len(ref.params) != len(cand.params):
        raise ValueError(f"argument count differs: reference has {len(ref.params)}, candidate {len(cand.params)}")

    ptr_sizes = _parse_kv(pointer_sizes)
    scalars = _parse_kv(scalar_values)
    warnings: list[str] = []
    outputs = _written_params(ref) | _written_params(cand)

    same_name = ref_func == cand_func
    ref_call = f"ref_{ref_func}" if same_name else ref_func

    decls, fills, ref_args, cand_args, checks = [], [], [], [], []
    for i, (p, q) in enumerate(zip(ref.params, cand.params)):
        t = p.ctype or "float"
        is_float = t in _FLOAT_TYPES or "fixed" in t
        if p.is_array:
            if p.dims and all(d.strip() for d in p.dims):
                dims = "".join(f"[{d}]" for d in p.dims)
            elif p.pointer or (p.dims and not p.dims[0].strip()):
                size = ptr_sizes.get(p.name) or ptr_sizes.get(q.name)
                if not size:
                    size = "1024"
                    warnings.append(f"Size of pointer argument '{p.name}' unknown; assumed 1024 elements. Pass pointer_sizes='{p.name}=<n>'.")
                inner = "".join(f"[{d}]" for d in p.dims[1:])
                dims = f"[{size}]{inner}"
            decls.append(f"static {t} ref_{p.name}{dims};\nstatic {t} cand_{p.name}{dims};")
            gen = "tb_rnd_real()" if is_float else "(" + t + ")(tb_rng() % 16)"
            fills.append(
                f"  {{ {t}* tb_p = ({t}*)ref_{p.name}; for (size_t tb_i = 0; tb_i < sizeof(ref_{p.name})/sizeof({t}); ++tb_i) tb_p[tb_i] = {gen}; "
                f"std::memcpy(cand_{p.name}, ref_{p.name}, sizeof(ref_{p.name})); }}"
            )
            cast_r = f"ref_{p.name}"
            cast_c = f"cand_{p.name}"
            if q.pointer and not q.dims and p.dims:
                cast_c = f"({q.ctype}*)cand_{p.name}"
            if p.pointer and not p.dims and q.dims:
                cast_r = f"({p.ctype}*)ref_{p.name}"
            ref_args.append(cast_r)
            cand_args.append(cast_c)
            if p.name in outputs or q.name in outputs:
                checks.append(
                    f"  tb_errors += tb_compare(\"{p.name}\", (const {t}*)ref_{p.name}, (const {t}*)cand_{p.name}, sizeof(ref_{p.name})/sizeof({t}), {'true' if is_float else 'false'});"
                )
        else:
            val = scalars.get(p.name) or scalars.get(q.name)
            if val is None:
                val = "0"
                warnings.append(f"Scalar argument '{p.name}' has no value; using 0. Pass scalar_values='{p.name}=<v>'.")
            decls.append(f"static const {t} arg_{p.name} = ({t})({val});")
            ref_args.append(f"arg_{p.name}")
            cand_args.append(f"arg_{p.name}")
    if not checks:
        warnings.append("No written array arguments detected; the testbench only checks that both versions run.")

    rename = (
        f"#define {ref_func} ref_{ref_func}\n#include \"{ref_include}\"\n#undef {ref_func}\n"
        if same_name
        else f"#include \"{ref_include}\"\n"
    )
    # Helpers come BEFORE the kernel includes so that kernel macros (e.g. `#define T 32`)
    # cannot rewrite them; every testbench identifier is tb_-prefixed for the same reason.
    tb = f"""// Auto-generated equivalence testbench: {ref_func} (reference) vs {cand_func} (HLS candidate)
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <random>

static std::mt19937 tb_rng({seed});
static float tb_rnd_real() {{ return std::uniform_real_distribution<float>(-1.0f, 1.0f)(tb_rng); }}

template <typename TbElem>
static int tb_compare(const char* tb_name, const TbElem* tb_ref, const TbElem* tb_cand, size_t tb_n, bool tb_is_float) {{
  int tb_errs = 0;
  double tb_max = 0.0;
  for (size_t tb_i = 0; tb_i < tb_n; ++tb_i) {{
    double tb_r = (double)tb_ref[tb_i], tb_c = (double)tb_cand[tb_i];
    double tb_err = std::fabs(tb_r - tb_c);
    double tb_tol = tb_is_float ? {tolerance} * (1.0 + std::fabs(tb_r)) : 0.0;
    if (tb_err > tb_max) tb_max = tb_err;
    if (!(tb_err <= tb_tol)) {{
      if (tb_errs < 5) std::printf("MISMATCH %s[%zu]: ref=%.8g cand=%.8g\\n", tb_name, tb_i, tb_r, tb_c);
      ++tb_errs;
    }}
  }}
  std::printf("%s: %zu elements, %d mismatches, max abs error %.3g\\n", tb_name, tb_n, tb_errs, tb_max);
  return tb_errs;
}}

{rename}#include "{cand_include}"

{chr(10).join(decls)}

int main() {{
{chr(10).join(fills)}
  {ref_call}({", ".join(ref_args)});
  {cand_func}({", ".join(cand_args)});
  int tb_errors = 0;
{chr(10).join(checks)}
  if (tb_errors) {{ std::printf("FAIL: %d mismatches\\n", tb_errors); return 1; }}
  std::printf("PASS\\n");
  return 0;
}}
"""
    return tb, warnings


def compile_and_run_files(sources: list[Path], timeout_s: int = 120, extra_flags: list[str] | None = None) -> dict:
    cxx = find_compiler()
    if not cxx:
        return {"status": "not_available", "message": "No C++ compiler found. Install g++/clang++ or `pip install ziglang`."}
    build = workspace_dir() / "build"
    build.mkdir(exist_ok=True)
    h = hashlib.sha1("|".join(str(s) for s in sources).encode()).hexdigest()[:10]
    exe = build / (f"sim_{h}.exe" if os.name == "nt" else f"sim_{h}")
    flags = ["-std=c++17", "-O2", "-w", "-I", str(workspace_dir())]
    for s in sources:
        flags += ["-I", str(s.parent)]
    if os.getenv("XILINX_HLS"):
        flags += ["-I", str(Path(os.environ["XILINX_HLS"]) / "include")]
    # Compile .c files as C++ so they share linkage with .cpp candidates.
    srcs: list[str] = []
    for s in sources:
        if s.suffix == ".c":
            srcs += ["-x", "c++", str(s), "-x", "none"]
        else:
            srcs.append(str(s))
    cmd = cxx + flags + (extra_flags or []) + srcs + ["-o", str(exe)]
    try:
        comp = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as e:
        return {"status": "error", "stage": "compile", "error": str(e)}
    if comp.returncode != 0:
        return {"status": "compile_error", "stage": "compile", "stderr": comp.stderr[-5000:], "command": " ".join(cmd)}
    try:
        run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "stage": "run", "error": f"simulation exceeded {timeout_s}s"}
    return {
        "status": "pass" if run.returncode == 0 else "fail",
        "returncode": run.returncode,
        "stdout": run.stdout[-5000:],
        "stderr": run.stderr[-2000:],
        "compiler": " ".join(cxx),
    }


# ---------------------------------------------------------------- ADK tools


def generate_testbench(
    reference_path: str,
    reference_function: str,
    candidate_path: str,
    candidate_function: str,
    testbench_path: str = "",
    pointer_sizes: str = "",
    scalar_values: str = "",
) -> dict:
    """Generate a C++ equivalence testbench that runs the original kernel and the
    HLS candidate on identical random inputs and compares every array they write.

    Args:
        reference_path: Original kernel source (workspace or project relative).
        reference_function: Function name in the original kernel.
        candidate_path: HLS candidate source (workspace relative).
        candidate_function: Top function name in the candidate.
        testbench_path: Where to write the testbench ("" = <candidate>_tb.cpp next to it).
        pointer_sizes: Element counts for pointer arguments, e.g. "A=4096,B=4096" ("" if none).
        scalar_values: Values for scalar arguments, e.g. "n=64,alpha=1.5" ("" if none).

    Returns:
        dict with `testbench_path` and `warnings`. Run it with run_testbench.
    """
    try:
        ref_p = resolve_read_path(reference_path)
        cand_p = resolve_read_path(candidate_path)
        tb_path = resolve_write_path(testbench_path or str(cand_p.with_name(cand_p.stem + "_tb.cpp")))
        tb, warnings = build_testbench(
            ref_p.read_text(encoding="utf-8", errors="replace"),
            os.path.relpath(ref_p, tb_path.parent).replace("\\", "/"),
            reference_function,
            cand_p.read_text(encoding="utf-8", errors="replace"),
            os.path.relpath(cand_p, tb_path.parent).replace("\\", "/"),
            candidate_function,
            pointer_sizes,
            scalar_values,
        )
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}
    tb_path.parent.mkdir(parents=True, exist_ok=True)
    tb_path.write_text(tb, encoding="utf-8")
    return {
        "status": "ok",
        "testbench_path": tb_path.relative_to(workspace_dir()).as_posix(),
        "warnings": warnings,
    }


def run_testbench(testbench_path: str, timeout_s: int = 120) -> dict:
    """Compile and run a testbench (C-simulation). A generated testbench already
    #includes the reference and candidate sources.

    Args:
        testbench_path: Testbench .cpp file (workspace relative).
        timeout_s: Max seconds for the simulation run.

    Returns:
        dict with `status` = pass / fail / compile_error / timeout, plus stdout/stderr.
    """
    try:
        tb = resolve_read_path(testbench_path)
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}
    return compile_and_run_files([tb], timeout_s=timeout_s)


def compile_and_run(source_paths: list[str], timeout_s: int = 120) -> dict:
    """Compile several C/C++ files into one program (one must define main) and run it.

    Args:
        source_paths: Files relative to the workspace or project root.
        timeout_s: Max seconds for the run.

    Returns:
        dict with `status` = pass / fail / compile_error / timeout, plus stdout/stderr.
    """
    try:
        paths = [resolve_read_path(p) for p in source_paths]
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}
    return compile_and_run_files(paths, timeout_s=timeout_s)
