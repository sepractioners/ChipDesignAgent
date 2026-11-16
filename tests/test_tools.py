"""Tool-level tests (no LLM needed)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from chip_design_agent.tools import (
    analyze_kernel,
    generate_testbench,
    lint_hls,
    list_files,
    list_optimizers,
    read_file,
    run_autosa,
    run_testbench,
    suggest_directives,
    write_file,
)
from chip_design_agent.tools.verification import find_compiler

ROOT = Path(__file__).resolve().parents[1]
FIX = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(autouse=True)
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("CHIP_AGENT_WORKSPACE", str(tmp_path))
    for var in ("AUTOSA_ROOT", "HARP_CMD", "PROMETHEUS_CMD", "LIFT_CMD"):
        monkeypatch.delenv(var, raising=False)
    shutil.copy(ROOT / "examples" / "gemm.c", tmp_path / "gemm.c")
    shutil.copy(ROOT / "examples" / "jacobi2d.c", tmp_path / "jacobi2d.c")
    for f in FIX.glob("*.cpp"):
        shutil.copy(f, tmp_path / f.name)
    return tmp_path


# ---------------------------------------------------------------- profiling


def test_gemm_is_detected_and_ranked():
    r = analyze_kernel("gemm.c")
    assert r["status"] == "ok"
    hot = r["loop_nests"][0]
    assert hot["pattern"]["pattern"] == "gemm"
    assert hot["pattern"]["dims"] == {"M": 64, "N": 64, "K": 64}
    assert hot["pattern"]["reduction_var"] == "k"
    assert hot["total_iterations"] == 64**3
    assert hot["share_of_ops"] > 0.99  # the C[i][j]=0 nest is negligible


def test_stencil_is_detected():
    r = analyze_kernel("jacobi2d.c")
    pat = r["loop_nests"][0]["pattern"]
    assert pat["pattern"] == "stencil"
    assert pat["radius"] == 1
    assert len(pat["points"]) == 5
    assert pat["in_place"] is False
    assert r["loop_nests"][0]["total_iterations"] == 126 * 126


def test_project_examples_are_readable():
    assert read_file("examples/gemm.c")["status"] == "ok"


# ---------------------------------------------------------------- transform


def test_lint_flags_non_synthesizable_code():
    shutil.copy(ROOT / "examples" / "vadd_bad.cpp", Path(list_files()["workspace"]) / "vadd_bad.cpp")
    r = lint_hls("vadd_bad.cpp", "vadd")
    assert not r["synthesizable"]
    msgs = " ".join(i["message"] for i in r["issues"])
    assert "Dynamic memory" in msgs and "STL" in msgs and "recursive" in msgs and "INTERFACE" in msgs


def test_lint_accepts_hls_kernel():
    r = lint_hls("gemm_hls.cpp", "gemm_hls")
    assert r["synthesizable"], r["issues"]
    assert r["pragmas_used"]["INTERFACE"] == 4


def test_writes_are_confined_to_workspace():
    assert write_file("../escape.txt", "x")["status"] == "error"
    assert write_file("sub/ok.txt", "x")["status"] == "ok"


def test_write_file_decodes_escaped_llm_output():
    escaped = (FIX / "gemma_escaped.txt").read_text()
    assert "\n" not in escaped.strip()  # one line of literal \n escapes
    r = write_file("gemm_fixed.cpp", escaped)
    assert r["status"] == "ok" and "note" in r
    assert r["lines"] > 20
    assert read_file("gemm_fixed.cpp")["content"].startswith("// Dense matrix multiply")
    assert lint_hls("gemm_fixed.cpp", "gemm")["functions"] == ["gemm"]


@pytest.mark.skipif(find_compiler() is None, reason="no C++ compiler")
def test_decoded_llm_output_passes_csim():
    write_file("gemm_fixed.cpp", (FIX / "gemma_escaped.txt").read_text())
    tb = generate_testbench("gemm.c", "gemm", "gemm_fixed.cpp", "gemm", testbench_path="fixed_tb.cpp")
    assert tb["status"] == "ok", tb
    r = run_testbench(tb["testbench_path"])
    assert r["status"] == "pass", r


def test_write_file_keeps_normal_source():
    src = 'void f() {\n  printf("a\\n");\n}\n'
    r = write_file("plain.cpp", src)
    assert "note" not in r
    assert read_file("plain.cpp")["content"] == src


def test_lint_reports_unreadable_file():
    shutil.copy(FIX / "gemma_escaped.txt", Path(list_files()["workspace"]) / "escaped.cpp")
    r = lint_hls("escaped.cpp", "gemm")
    assert not r["synthesizable"]
    assert any("No C/C++ function" in i["message"] for i in r["issues"])


def test_lint_catches_gemma_v1_hardware_mistakes():
    shutil.copy(FIX / "gemma_v1.cpp", Path(list_files()["workspace"]) / "gemma_v1.cpp")
    msgs = " ".join(i["message"] for i in lint_hls("gemma_v1.cpp", "gemm")["issues"])
    assert "depth=1024 but the array has 4096 elements" in msgs
    assert "cannot be partitioned" in msgs
    assert "fully unrolls every loop" in msgs


# ------------------------------------------------------------- optimization


def test_suggestions_for_gemm_include_autosa_sizes():
    s = suggest_directives("gemm.c")["suggestions"][0]
    assert s["pattern"] == "gemm"
    assert "array_part[32,32,32]" in s["autosa_sa_sizes"]


def test_external_optimizers_report_not_available():
    assert not any(o["available"] for o in list_optimizers()["optimizers"].values())
    assert run_autosa("gemm.c", "{}")["status"] == "not_available"


# ------------------------------------------------------------- verification

needs_cxx = pytest.mark.skipif(find_compiler() is None, reason="no C++ compiler")


@needs_cxx
def test_correct_hls_version_passes_csim():
    tb = generate_testbench("gemm.c", "gemm", "gemm_hls.cpp", "gemm_hls")
    assert tb["status"] == "ok", tb
    r = run_testbench(tb["testbench_path"])
    assert r["status"] == "pass", r
    assert "PASS" in r["stdout"]


@needs_cxx
def test_kernel_macro_named_T_does_not_break_testbench_and_real_bug_is_caught():
    # Gemma 4's tiled GEMM: `#define T 32`, and it loads C instead of zeroing it on the first k tile.
    tb = generate_testbench("gemm.c", "gemm", "gemma_v3_tiled.cpp", "gemm_hls", testbench_path="tiled_tb.cpp")
    r = run_testbench(tb["testbench_path"])
    assert r["status"] == "fail", r  # compiles now, and the logic bug shows up
    assert "MISMATCH C[" in r["stdout"]


def test_lint_flags_pragma_placed_before_loop():
    r = lint_hls("gemma_v3_tiled.cpp", "gemm_hls")
    misplaced = [i for i in r["issues"] if "ENCLOSING loop" in i["message"]]
    assert {i["line"] for i in misplaced} == {49, 51, 56}
    assert lint_hls("gemm_hls.cpp", "gemm_hls")["warnings"] == 0  # correctly placed pragmas stay clean


@needs_cxx
def test_corrected_tiled_version_passes():
    src = (FIX / "gemma_v3_tiled.cpp").read_text()
    old = "C_local[i][j] = C[m_start + i][n_start + j];"
    assert old in src
    write_file("tiled_ok.cpp", src.replace(old, "C_local[i][j] = (k_start == 0) ? 0.0f : C[m_start + i][n_start + j];"))
    tb = generate_testbench("gemm.c", "gemm", "tiled_ok.cpp", "gemm_hls", testbench_path="tiled_ok_tb.cpp")
    r = run_testbench(tb["testbench_path"])
    assert r["status"] == "pass", r


@needs_cxx
def test_buggy_hls_version_fails_csim_even_with_same_function_name():
    tb = generate_testbench("gemm.c", "gemm", "gemm_hls_buggy.cpp", "gemm", testbench_path="buggy_tb.cpp")
    assert tb["status"] == "ok", tb
    r = run_testbench(tb["testbench_path"])
    assert r["status"] == "fail", r
    assert "MISMATCH C[" in r["stdout"]
