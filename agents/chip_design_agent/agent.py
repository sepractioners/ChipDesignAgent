"""Agentic SW -> HLS flow (Google ADK).

                 +-------------+
   user  ----->  | orchestrator| <---> profiling_agent     (bottlenecks, patterns)
                 +-------------+ <---> transform_agent     (SW -> HLS C++)
                                 <---> optimization_agent  (AutoSA/HARP/Prometheus/LIFT, pragmas)
                                 <---> verification_agent  (testbench, C-sim)

The specialists are wrapped as AgentTools, so the orchestrator stays in
control, gets each specialist's report back, and can loop (e.g. send a failing
C-sim back to the transform agent).
"""

from __future__ import annotations

from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool

from .config import get_model
from .tools import (
    analyze_kernel,
    compile_and_run,
    generate_testbench,
    hls_pragma_reference,
    lint_hls,
    list_files,
    list_optimizers,
    read_file,
    run_autosa,
    run_external_optimizer,
    run_hls_synthesis,
    run_testbench,
    suggest_directives,
    write_file,
)

profiling_agent = LlmAgent(
    name="profiling_agent",
    model=get_model("profiling"),
    description="Profiles a C/C++ kernel: ranks loop nests by cost (bottleneck analysis) and classifies their pattern (gemm, stencil, reduction, elementwise).",
    instruction="""You are the profiling specialist in a hardware-acceleration flow.
1. Call analyze_kernel on the file you are given.
2. Report, concisely:
   - the kernel function(s) and their arguments,
   - the hottest loop nest(s): function, line, trip counts, estimated ops and share of total,
   - the matched pattern of each hot nest and the recommended optimizers,
   - anything that will make HLS hard (unknown loop bounds, pointer arguments with unknown size).
Only report facts from the tool output. Do not write code.""",
    tools=[analyze_kernel, read_file, list_files],
)

transform_agent = LlmAgent(
    name="transform_agent",
    model=get_model("transform"),
    description="Rewrites a software kernel into synthesizable Vitis HLS C++ and fixes lint or simulation failures.",
    instruction="""You are the SW->HLS code transformation specialist.
Given a source kernel (and optionally profiling notes, optimization directives or an error report):
1. read_file the source. Call hls_pragma_reference if you need pragma syntax.
2. Write a synthesizable HLS C++ version with write_file. Rules:
   - Keep the SAME function signature (argument order and types) so it can be tested against the original.
     Name the function <original_name>_hls unless told otherwise.
   - #include the original's #define constants by copying them (do not #include the original file).
   - No malloc/new, recursion, STL containers or I/O. Fixed loop bounds.
   - Add INTERFACE pragmas on the top function, then apply the requested optimizations
     (PIPELINE, UNROLL + ARRAY_PARTITION, tiling, local buffers, line buffers, partial sums).
   - Keep the arithmetic order of floating-point reductions unless told that small differences are fine.
   - Call write_file with the plain source text as `content`: real line breaks, not a quoted or escaped string.
   - Do not #include vendor headers (ap_int.h, hls_stream.h) unless you actually use their types.
3. Call lint_hls on your file with the top function name. Fix every `error` and every `warning` about
   m_axi depth, partitioning or PIPELINE placement, then lint again until it is clean.
4. Reply with: the file path, the top function name, and a short list of the transformations you applied.""",
    tools=[read_file, write_file, lint_hls, hls_pragma_reference, list_files],
)

optimization_agent = LlmAgent(
    name="optimization_agent",
    model=get_model("optimization"),
    description="Chooses and runs optimizers (AutoSA, HARP, Prometheus, LIFT or the built-in pattern rules) and returns concrete HLS directives / restructuring; can run HLS synthesis to measure QoR.",
    instruction="""You are the optimization specialist.
1. Call list_optimizers to see which back-ends are configured.
2. Call suggest_directives on the kernel to get pattern-based restructuring, pragmas and (for gemm) AutoSA sa_sizes.
3. If a configured external optimizer fits the pattern (gemm -> autosa; stencil -> prometheus or lift; pragma tuning -> harp),
   run it (run_autosa / run_external_optimizer) and summarize its output files.
   If it reports not_available, say so in one line and continue with the built-in suggestions.
4. If asked to measure an HLS file and synthesis is available, call run_hls_synthesis and report latency, interval and resources.
Reply with a numbered, concrete optimization plan the transform agent can apply directly (exact pragmas and where they go).""",
    tools=[list_optimizers, suggest_directives, run_autosa, run_external_optimizer, run_hls_synthesis, read_file, write_file],
)

verification_agent = LlmAgent(
    name="verification_agent",
    model=get_model("verification"),
    description="Generates an equivalence testbench (original vs HLS candidate) and runs C-simulation, reporting PASS/FAIL with mismatch details.",
    instruction="""You are the testing/verification specialist.
Given the original kernel file + function and the HLS candidate file + function:
1. Call generate_testbench. If the original has pointer arguments, pass pointer_sizes (e.g. "A=4096"),
   and pass scalar_values for scalar arguments, using sizes from the source's #defines.
2. Call run_testbench on the generated testbench.
3. If status is compile_error, report the first compiler errors verbatim (they are usually in the candidate).
   If fail, report the mismatching arrays and values.
4. Reply starting with exactly one of: VERIFIED, FAILED, or BLOCKED (tool missing), then the details.""",
    tools=[generate_testbench, run_testbench, compile_and_run, read_file, list_files],
)

root_agent = LlmAgent(
    name="chip_design_orchestrator",
    model=get_model("orchestrator"),
    description="Orchestrates profiling, SW->HLS transformation, optimization and verification of a kernel.",
    instruction="""You orchestrate an agentic flow that turns a software kernel (C/C++) into an optimized, verified HLS design.
You do not write code yourself; you delegate to your specialist tools and pass them precise requests
(file paths, function names, and the relevant findings from earlier steps).

Workflow:
1. If no kernel file is named, call list_files and ask the user which kernel to accelerate.
2. profiling_agent: profile the kernel.
3. optimization_agent: get an optimization plan for the hottest nest(s), given the pattern.
4. transform_agent: write the HLS version, applying the plan. Get back the file path and top function.
5. verification_agent: check the HLS version against the original.
6. If FAILED: send the exact error report back to transform_agent to fix the HLS file, then verify again.
   Do at most 3 fix attempts. If BLOCKED, report what is missing.
7. Optionally, if synthesis is available, ask optimization_agent to synthesize the verified file and,
   if the user wants, do one more optimize -> transform -> verify round.

Finish with a short report: hot spots and pattern, optimizations applied, files written, verification result,
and recommended next steps.""",
    tools=[
        AgentTool(agent=profiling_agent),
        AgentTool(agent=optimization_agent),
        AgentTool(agent=transform_agent),
        AgentTool(agent=verification_agent),
        list_files,
        read_file,
    ],
)
