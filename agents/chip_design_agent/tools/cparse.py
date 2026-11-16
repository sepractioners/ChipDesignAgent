"""Lightweight C/C++ kernel parser.

This is deliberately not a full C front end: it understands the shapes that
matter for HLS kernels (functions, `#define` constants, counted `for` loop
nests, array accesses) and degrades gracefully on everything else. Swap in
libclang or MLIR later if you need precision.
"""

from __future__ import annotations

import ast
import operator
import re
from dataclasses import dataclass, field

# ------------------------------------------------------------------ basics


def strip_comments(src: str) -> str:
    # keep newlines so line numbers stay correct after multi-line comments
    src = re.sub(r"/\*.*?\*/", lambda m: re.sub(r"[^\n]", " ", m.group(0)), src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def parse_defines(src: str) -> dict[str, int | float]:
    """Numeric `#define NAME value` constants (values may reference earlier ones)."""
    defines: dict[str, int | float] = {}
    for m in re.finditer(r"^\s*#\s*define\s+(\w+)\s+(.+?)\s*$", src, flags=re.M):
        val = eval_expr(m.group(2), defines)
        if val is not None:
            defines[m.group(1)] = val
    return defines


_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Div: operator.floordiv,  # C integer division
    ast.Mod: operator.mod,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}


def eval_expr(expr: str, env: dict[str, int | float]) -> int | float | None:
    """Safely evaluate a constant C integer expression, or return None."""
    expr = re.sub(r"(\d+)[uUlL]+\b", r"\1", expr.strip())
    expr = re.sub(r"\(\s*(?:int|unsigned|long|size_t)\s*\)", "", expr)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return None

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.Name):
            if n.id in env:
                return env[n.id]
            raise ValueError(n.id)
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
            return _BINOPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -ev(n.operand)
        raise ValueError(ast.dump(n))

    try:
        return ev(tree)
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def match_brace(src: str, open_idx: int) -> int:
    """Index of the `}` matching the `{` at open_idx (or len(src)-1)."""
    depth = 0
    for i in range(open_idx, len(src)):
        c = src[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
    return len(src) - 1


def match_paren(src: str, open_idx: int) -> int:
    depth = 0
    for i in range(open_idx, len(src)):
        if src[i] == "(":
            depth += 1
        elif src[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return len(src) - 1


# --------------------------------------------------------------- functions


@dataclass
class Param:
    ctype: str
    name: str
    dims: list[str] = field(default_factory=list)  # array dims as written
    pointer: bool = False

    @property
    def is_array(self) -> bool:
        return bool(self.dims) or self.pointer


@dataclass
class Function:
    name: str
    return_type: str
    params: list[Param]
    body: str  # text between the braces
    body_offset: int


_FUNC_RE = re.compile(
    r"(?P<ret>\b[A-Za-z_][\w:<>, \t\*&]*?)[ \t\*&]+(?P<name>[A-Za-z_]\w*)\s*\("
)
_KEYWORDS = {"if", "for", "while", "switch", "return", "sizeof", "else", "do"}


def parse_functions(src: str) -> list[Function]:
    src = strip_comments(src)
    funcs: list[Function] = []
    pos = 0
    while True:
        m = _FUNC_RE.search(src, pos)
        if not m:
            break
        name = m.group("name")
        lp = m.end() - 1
        rp = match_paren(src, lp)
        after = src[rp + 1 :].lstrip()
        ret = m.group("ret").strip().split()[-1] if m.group("ret").strip() else ""
        if name in _KEYWORDS or ret in _KEYWORDS or not after.startswith("{"):
            pos = m.end()
            continue
        # skip things like `else if (...) {`
        line_start = src.rfind("\n", 0, m.start()) + 1
        if src[line_start : m.start()].strip().startswith("#"):
            pos = m.end()
            continue
        brace = src.index("{", rp)
        end = match_brace(src, brace)
        funcs.append(
            Function(
                name=name,
                return_type=" ".join(m.group("ret").split()),
                params=parse_params(src[lp + 1 : rp]),
                body=src[brace + 1 : end],
                body_offset=brace + 1,
            )
        )
        pos = end + 1
    return funcs


def parse_params(text: str) -> list[Param]:
    params: list[Param] = []
    text = text.strip()
    if not text or text == "void":
        return params
    depth, cur, parts = 0, "", []
    for c in text:  # split on top-level commas only
        if c in "<([":
            depth += 1
        elif c in ">)]":
            depth -= 1
        if c == "," and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += c
    parts.append(cur)
    for part in parts:
        part = part.strip()
        dims = re.findall(r"\[([^\]]*)\]", part)
        decl = re.sub(r"\[[^\]]*\]", "", part).strip()
        m = re.match(r"(.*?)([A-Za-z_]\w*)\s*$", decl)
        if not m:
            continue
        ctype = m.group(1).strip()
        pointer = "*" in ctype
        ctype = ctype.replace("*", "").replace("&", "").strip()
        ctype = re.sub(r"\b(const|volatile|restrict|__restrict__)\b", "", ctype).strip()
        params.append(Param(ctype=" ".join(ctype.split()), name=m.group(2), dims=dims, pointer=pointer))
    return params


# ------------------------------------------------------------------- loops


@dataclass
class Loop:
    var: str
    lower: str
    upper: str
    cmp: str
    step: str
    trip_count: int | None
    body: str
    children: list["Loop"] = field(default_factory=list)
    label: str = ""
    line: int = 0

    def own_statements(self) -> list[str]:
        """Statements directly in this loop's body (nested loops removed)."""
        text = self.body
        for child in self.children:
            text = text.replace(child._full_text, " ", 1)
        stmts = [s.strip() for s in re.split(r";", text)]
        return [s.strip("{} \n\t") for s in stmts if s.strip("{} \n\t") and not s.strip().startswith("#")]

    _full_text: str = ""


_FOR_RE = re.compile(r"\bfor\s*\(")
_HEADER_RE = re.compile(
    r"^\s*(?:(?:const\s+)?(?:unsigned\s+|signed\s+)?(?:int|long|short|size_t|unsigned|char|auto|ap_uint<\d+>|ap_int<\d+>)\s+)?"
    r"(?P<var>\w+)\s*=\s*(?P<lo>[^;]+);"
    r"\s*(?P<lhs>[^;<>=!]+?)\s*(?P<cmp><=|<|>=|>|!=)\s*(?P<hi>[^;]+);"
    r"\s*(?P<step>[^)]*)$",
    re.S,
)


def _statement_end(src: str, start: int) -> int:
    """End index (exclusive) of the C statement beginning at `start`."""
    i = start
    while i < len(src) and src[i].isspace():
        i += 1
    if i >= len(src):
        return i
    if src[i] == "{":
        return match_brace(src, i) + 1
    m = re.match(r"(for|while|if)\s*\(", src[i:])
    if m:
        rp = match_paren(src, i + m.end() - 1)
        end = _statement_end(src, rp + 1)
        if m.group(1) == "if":
            em = re.match(r"\s*else\b", src[end:])
            if em:
                end = _statement_end(src, end + em.end())
        return end
    semi = src.find(";", i)
    return len(src) if semi < 0 else semi + 1


def parse_loops(src: str, defines: dict[str, int | float], base_line: int = 1) -> list[Loop]:
    loops: list[Loop] = []
    pos = 0
    while True:
        m = _FOR_RE.search(src, pos)
        if not m:
            break
        lp = m.end() - 1
        rp = match_paren(src, lp)
        header = src[lp + 1 : rp]
        body_start = rp + 1
        body_end = _statement_end(src, body_start)
        body = src[body_start:body_end].strip()
        if body.startswith("{") and body.endswith("}"):
            body = body[1:-1]
        label_m = re.search(r"(\w+)\s*:\s*$", src[: m.start()])
        line = base_line + src.count("\n", 0, m.start())
        loop = _make_loop(header, body, defines, label_m.group(1) if label_m else "", line)
        loop._full_text = src[m.start() : body_end]
        body_line = base_line + src.count("\n", 0, body_start)
        loop.children = parse_loops(src[body_start:body_end], defines, body_line)
        loops.append(loop)
        pos = body_end
    return loops


def _make_loop(header: str, body: str, defines, label: str, line: int) -> Loop:
    h = _HEADER_RE.match(header)
    if not h:
        return Loop("?", "?", "?", "?", "?", None, body, label=label, line=line)
    var, lo, hi, cmp, step = h["var"], h["lo"].strip(), h["hi"].strip(), h["cmp"], h["step"].strip()
    lo_v, hi_v = eval_expr(lo, defines), eval_expr(hi, defines)
    stride = _step_size(step, var, defines)
    trip = None
    if lo_v is not None and hi_v is not None and stride:
        if cmp in ("<", "!="):
            trip = max(0, -(-(hi_v - lo_v) // stride))
        elif cmp == "<=":
            trip = max(0, (hi_v - lo_v) // stride + 1)
        elif cmp == ">":
            trip = max(0, -(-(lo_v - hi_v) // -stride)) if stride < 0 else None
        elif cmp == ">=":
            trip = max(0, (lo_v - hi_v) // -stride + 1) if stride < 0 else None
        trip = int(trip) if trip is not None else None
    return Loop(var, lo, hi, cmp, step, trip, body, label=label, line=line)


def _step_size(step: str, var: str, defines) -> int | None:
    s = step.replace(" ", "")
    if s in (f"{var}++", f"++{var}"):
        return 1
    if s in (f"{var}--", f"--{var}"):
        return -1
    m = re.fullmatch(rf"{var}(\+=|-=)(.+)", s) or re.fullmatch(rf"{var}=({var}[+-])(.+)", s)
    if m:
        v = eval_expr(m.group(2), defines)
        if v is None:
            return None
        return int(v) if "+" in m.group(1) else -int(v)
    return None


# ---------------------------------------------------------- array accesses


@dataclass
class Access:
    array: str
    indices: list[str]


_ACCESS_RE = re.compile(r"\b([A-Za-z_]\w*)\s*((?:\[[^\[\]]+\]\s*)+)")


def find_accesses(expr: str) -> list[Access]:
    out = []
    for m in _ACCESS_RE.finditer(expr):
        idx = [i.strip() for i in re.findall(r"\[([^\[\]]+)\]", m.group(2))]
        out.append(Access(m.group(1), idx))
    return out


_ASSIGN_RE = re.compile(r"^(?P<lhs>.+?)\s*(?P<op>\+=|-=|\*=|/=|(?<![=!<>])=(?!=))\s*(?P<rhs>.+)$", re.S)


def split_assignment(stmt: str):
    m = _ASSIGN_RE.match(stmt.strip())
    if not m:
        return None
    lhs = re.sub(r"^(?:const\s+)?(?:float|double|int|long|short|char|auto|unsigned)\s+", "", m["lhs"].strip())
    return lhs, m["op"], m["rhs"].strip()


def count_ops(expr: str) -> dict[str, int]:
    e = re.sub(r"\[[^\[\]]*\]", "", expr)  # ignore index arithmetic
    return {
        "add": len(re.findall(r"(?<![+\-eE])[+-](?![+\-=])", e)),
        "mul": e.count("*"),
        "div": e.count("/"),
    }


def offset_of(index: str, var: str) -> int | None:
    """If index is `var`, `var+c` or `var-c` return c, else None."""
    s = index.replace(" ", "")
    if s == var:
        return 0
    m = re.fullmatch(rf"{re.escape(var)}([+-]\d+)", s) or re.fullmatch(rf"(\d+)\+{re.escape(var)}", s)
    return int(m.group(1)) if m else None


def index_vars(index: str) -> set[str]:
    return set(re.findall(r"[A-Za-z_]\w*", index))
