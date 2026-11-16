# ChipDesignAgent

A multi-agent **software → hardware (HLS)** flow built on [Google ADK](https://google.github.io/adk-docs/),
running on a local **Ollama `gemma4`** model by default.

The architecture follows the *"Outlook: An Agentic Approach"* slide from the DAC 62 Day 3 keynote,
**"Democratize Chip Design — with Deep Learning and Automated Code Transformation"**. A central
orchestrator takes user code (C/C++; PyTorch/MLIR front ends are future work) and iterates over four
specialist agents.

```
                        ┌───────────────────────────┐
              ┌────────►│ profiling_agent           │  bottleneck analysis, pattern matching
              │         └───────────────────────────┘  (gemm / stencil / reduction / elementwise)
              │         ┌───────────────────────────┐
user ──► orchestrator ◄►│ optimization_agent        │  AutoSA · HARP · Prometheus · LIFT
              │         └───────────────────────────┘  + built-in pattern rules, HLS synthesis
              │         ┌───────────────────────────┐
              ├────────►│ transform_agent           │  SW → HLS C++, free-form rewrites, lint
              │         └───────────────────────────┘
              │         ┌───────────────────────────┐
              └────────►│ verification_agent        │  testbench generation, C-simulation
                        └───────────────────────────┘
```

The specialists are wired in as ADK `AgentTool`s, so the orchestrator gets each report back and can loop.
For example, a failing C-sim goes back to `transform_agent` (at most 3 fix attempts).

## What is real vs. pluggable

| Component | Status |
|---|---|
| Profiling (`analyze_kernel`) | Real. Static loop-nest analysis with trip counts, op counts and ranking; GEMM and stencil detection. Lightweight parser, not a full C front end. |
| Lint (`lint_hls`) | Real. Flags malloc/new, recursion, STL, I/O, missing INTERFACE pragmas, non-constant bounds. |
| Built-in optimizer (`suggest_directives`) | Real. Tiling, pipelining, partitioning, line buffers and partial sums, plus AutoSA `sa_sizes`. |
| Verification (`generate_testbench`, `run_testbench`) | Real. Equivalence C-sim of original vs HLS on random inputs. Uses g++/clang++, else the pip-installed `ziglang` compiler. |
| AutoSA (`run_autosa`) | Wrapper. Needs `AUTOSA_ROOT` (Linux/WSL install). |
| HARP / Prometheus / LIFT (`run_external_optimizer`) | Wrapper. Needs a `<NAME>_CMD` template; these are research artifacts without a standard CLI. |
| Vitis HLS synthesis (`run_hls_synthesis`) | Wrapper. Needs `vitis_hls` on PATH or `XILINX_HLS`. |

Any wrapper whose tool isn't configured returns `not_available`, and the agents fall back to the built-in rules.

## Setup

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -e .[dev]
ollama pull gemma4
```

The settings live in [agents/chip_design_agent/.env](agents/chip_design_agent/.env). `CHIP_AGENT_MODEL` picks the model:
`ollama_chat/gemma4` (default), `gemini-...`, or any LiteLLM name. `CHIP_AGENT_MODEL_<ROLE>` overrides the model for
one agent, e.g. a bigger model just for `TRANSFORM`. Use the `ollama_chat/` prefix; plain `ollama/` breaks tool calling.

## Run

ADK web UI, which shows the agent and tool trace:

```bash
.venv\Scripts\adk web agents
```

Command line:

```bash
.venv\Scripts\python run_flow.py examples/gemm.c
```

Tool tests (no LLM needed):

```bash
.venv\Scripts\python -m pytest -q
```

Generated files (HLS sources, testbenches, build outputs) go to `workspace/`.

## Layout

```
agents/chip_design_agent/
  agent.py            orchestrator + 4 specialist agents
  config.py           model selection (Ollama via LiteLLM, or Gemini)
  tools/
    cparse.py         lightweight C parser (functions, loops, accesses)
    profiling.py      analyze_kernel
    transform.py      lint_hls, hls_pragma_reference
    optimization.py   suggest_directives, run_autosa, run_external_optimizer, run_hls_synthesis
    verification.py   generate_testbench, run_testbench, compile_and_run
    files.py          workspace-sandboxed read/write
examples/             gemm.c (linear algebra), jacobi2d.c (stencil), vadd_bad.cpp (lint demo)
tests/                tool tests + HLS fixtures
run_flow.py           CLI runner
```

## Next steps

- PyTorch / MLIR front ends, e.g. `torch.export` → linalg, then pattern matching on linalg ops.
- Replace the regex parser with libclang or an MLIR affine analysis for precise dependences.
- Add QoR feedback to the loop: synthesis report → optimization agent → transform.
- Add a HARP-style learned cost model so the optimizer can explore without a full synthesis run.
