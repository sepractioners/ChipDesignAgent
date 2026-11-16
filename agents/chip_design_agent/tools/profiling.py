"""Profiling tools: bottleneck analysis + computation pattern matching."""

from __future__ import annotations

from collections import Counter

from . import cparse
from .files import resolve_read_path


def _walk(loop: cparse.Loop, ancestors: list[cparse.Loop]):
    yield loop, ancestors
    for c in loop.children:
        yield from _walk(c, ancestors + [loop])


def _nest_summary(chain: list[cparse.Loop]) -> dict:
    trips = [l.trip_count for l in chain]
    total = 1
    for t in trips:
        total = total * t if (t is not None and total is not None) else None
    inner = chain[-1]
    stmts = inner.own_statements()
    ops = Counter()
    reads = writes = 0
    for s in stmts:
        a = cparse.split_assignment(s)
        if a:
            lhs, op, rhs = a
            ops.update(cparse.count_ops(rhs))
            if op != "=":
                ops["add" if op in ("+=", "-=") else "mul"] += 1
            reads += len(cparse.find_accesses(rhs)) + (1 if op != "=" and cparse.find_accesses(lhs) else 0)
            writes += len(cparse.find_accesses(lhs))
        else:
            ops.update(cparse.count_ops(s))
    flops_per_iter = sum(ops.values())
    return {
        "loops": [
            {"var": l.var, "label": l.label, "line": l.line, "bounds": f"{l.lower} {l.cmp} {l.upper}", "trip_count": l.trip_count}
            for l in chain
        ],
        "depth": len(chain),
        "total_iterations": total,
        "ops_per_iteration": dict(ops),
        "mem_reads_per_iteration": reads,
        "mem_writes_per_iteration": writes,
        "estimated_ops": total * flops_per_iter if total is not None else None,
        "innermost_statements": stmts,
    }


def match_pattern(chain: list[cparse.Loop]) -> dict:
    """Classify the innermost statements of a loop nest."""
    loop_vars = [l.var for l in chain]
    for stmt in chain[-1].own_statements():
        a = cparse.split_assignment(stmt)
        if not a:
            continue
        lhs, op, rhs = a
        w = cparse.find_accesses(lhs)
        reads = cparse.find_accesses(rhs)
        # `C[i][j] = C[i][j] + ...` is an accumulation too
        accumulate = op in ("+=", "-=") or (w and rhs.replace(" ", "").startswith(lhs.replace(" ", "") + "+"))

        # ---- GEMM / matmul-like: W[i][j] += X[i][k] * Y[k][j]
        if accumulate and w and len(reads) >= 2 and "*" in rhs and len(chain) >= 3:
            w_vars = set().union(*(cparse.index_vars(i) for i in w[0].indices))
            others = [r for r in reads if r.array != w[0].array]
            if len(others) >= 2:
                r1, r2 = others[0], others[1]
                v1 = set().union(*(cparse.index_vars(i) for i in r1.indices))
                v2 = set().union(*(cparse.index_vars(i) for i in r2.indices))
                red = (v1 & v2) - w_vars
                if red and w_vars <= set(loop_vars) and (v1 | v2) >= w_vars:
                    k = sorted(red)[0]
                    trips = {l.var: l.trip_count for l in chain}
                    out_vars = [v for v in loop_vars if v in w_vars]
                    return {
                        "pattern": "gemm",
                        "description": "Matrix-multiply-like reduction (systolic array candidate)",
                        "output": w[0].array,
                        "inputs": [r1.array, r2.array],
                        "reduction_var": k,
                        "dims": {
                            "M": trips.get(out_vars[0]) if out_vars else None,
                            "N": trips.get(out_vars[1]) if len(out_vars) > 1 else None,
                            "K": trips.get(k),
                        },
                        "statement": stmt,
                        "recommended_optimizers": ["autosa", "harp", "prometheus"],
                    }

        # ---- Stencil: W[i][j] = f(A[i-1][j], A[i+1][j], A[i][j-1], ...)
        if w and len(reads) >= 3:
            by_array = Counter(r.array for r in reads)
            arr, n = by_array.most_common(1)[0]
            if n >= 3:
                acc = [r for r in reads if r.array == arr]
                offsets, ok = [], True
                for r in acc:
                    offs = []
                    for idx in r.indices:
                        var = next((v for v in loop_vars if cparse.offset_of(idx, v) is not None), None)
                        if var is None:
                            ok = False
                            break
                        offs.append(cparse.offset_of(idx, var))
                    if not ok:
                        break
                    offsets.append(tuple(offs))
                if ok and any(any(o) for o in offsets):
                    radius = max(abs(o) for off in offsets for o in off)
                    return {
                        "pattern": "stencil",
                        "description": f"{len(acc[0].indices)}-D stencil, {len(set(offsets))} points, radius {radius}",
                        "input": arr,
                        "output": w[0].array,
                        "points": sorted(set(offsets)),
                        "radius": radius,
                        "in_place": arr == w[0].array,
                        "statement": stmt,
                        "recommended_optimizers": ["prometheus", "line-buffer/sliding-window (built-in suggestion)"],
                    }

        if accumulate:
            return {
                "pattern": "reduction",
                "description": "Accumulation into " + (w[0].array if w else lhs),
                "statement": stmt,
                "recommended_optimizers": ["built-in: partial sums / tree reduction"],
            }
    stmts = chain[-1].own_statements()
    if stmts and all(cparse.split_assignment(s) for s in stmts):
        return {"pattern": "elementwise", "description": "Independent element-wise updates", "recommended_optimizers": ["built-in: pipeline II=1 + unroll"]}
    return {"pattern": "unknown", "description": "No known pattern matched", "recommended_optimizers": []}


def analyze_source(src: str) -> dict:
    clean = cparse.strip_comments(src)
    defines = cparse.parse_defines(clean)
    functions = []
    nests = []
    for fn in cparse.parse_functions(src):
        base_line = clean.count("\n", 0, fn.body_offset) + 1
        loops = cparse.parse_loops(fn.body, defines, base_line)
        functions.append(
            {
                "name": fn.name,
                "return_type": fn.return_type,
                "params": [
                    {"name": p.name, "type": p.ctype, "dims": p.dims, "pointer": p.pointer} for p in fn.params
                ],
                "top_level_loops": len(loops),
            }
        )
        for top in loops:
            for loop, anc in _walk(top, []):
                if loop.children:
                    continue  # analyse each nest at its innermost loop
                chain = anc + [loop]
                summary = _nest_summary(chain)
                summary["function"] = fn.name
                summary["pattern"] = match_pattern(chain)
                nests.append(summary)

    known = [n["estimated_ops"] for n in nests if n["estimated_ops"] is not None]
    total_ops = sum(known) if known else None
    nests.sort(key=lambda n: -(n["estimated_ops"] or 0))
    for rank, n in enumerate(nests, 1):
        n["rank"] = rank
        if total_ops and n["estimated_ops"] is not None:
            n["share_of_ops"] = round(n["estimated_ops"] / total_ops, 4)
    return {"defines": defines, "functions": functions, "loop_nests": nests, "total_estimated_ops": total_ops}


# ---------------------------------------------------------------- ADK tool


def analyze_kernel(path: str) -> dict:
    """Profile a C/C++ kernel statically: find loop nests, rank them by
    estimated operation count (bottleneck analysis), and classify each nest's
    computation pattern (gemm, stencil, reduction, elementwise).

    Args:
        path: Kernel source file, relative to the workspace or project root.

    Returns:
        dict with `functions`, `loop_nests` (ranked hottest first, each with
        trip counts, ops/iteration, memory accesses, and `pattern`), and a
        short `summary` string.
    """
    try:
        src = resolve_read_path(path).read_text(encoding="utf-8", errors="replace")
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "error": str(e)}
    result = analyze_source(src)
    if not result["loop_nests"]:
        result["summary"] = "No counted for-loops found."
    else:
        hot = result["loop_nests"][0]
        share = hot.get("share_of_ops")
        result["summary"] = (
            f"Hottest nest: {hot['function']} line {hot['loops'][0]['line']}, depth {hot['depth']}, "
            f"pattern={hot['pattern']['pattern']}, est. ops={hot['estimated_ops']}"
            + (f" ({share:.0%} of total)" if share is not None else "")
        )
    result["status"] = "ok"
    return result
