# normalizer.py
from __future__ import annotations

from pathlib import Path
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Callable

from problog.program import PrologString
from problog.logic import Term, Var, Constant, Clause, AnnotatedDisjunction, And, Or, Not

from problog import engine_builtin
from problog.logic import _arithmetic_functions 

from models.solve_path import PathSolver
from models.models import _unquote_functor
from .ast_structure import (
    # Statements:
    StatementBase, FactStmt, RuleStmt, ExternalStmt, ImportStmts, SrcLoc, 
    AgentStmt,

    # Expressions:
    Expr, VariableExpr, ConstantExpr, AtomExpr, TermExpr, ArithmeticExpr, ListExpr,
    LogicExpr, BuiltinExpr, IfThenExpr, IfThenElseExpr, TrueExpr, FailExpr, 
    AgentMarkExpr,

    ProgramIR, # the full program IR
)
from models.agent_structure import IRTDict, IRTMap
from .get_extern_arities import extern_arities

# ------------------------------------------------------------
# Small utilities
# ------------------------------------------------------------

def _sig(functor: str, arity: int) -> str:
    return f"{functor}/{arity}"

def _is_var_like(x: Any) -> bool:
    # ProbLog variables can be: Var("X"), int (internal), None (anonymous)
    return x is None or type(x) is int or isinstance(x, Var)

def _var_name(x: Any) -> str:
    if x is None:
        return "_"
    if type(x) is int:
        # follow problog.logic.term2str convention (common)
        return f"A{x+1}" if x >= 0 else f"X{-x}"
    return getattr(x, "name", str(x))

def _is_nil(t: Any) -> bool:
    return isinstance(t, Term) and (not _is_var_like(t)) and _unquote_functor(t.functor) == "[]" and t.arity == 0

def _is_list_cell(t: Any) -> bool:
    return isinstance(t, Term) and (not _is_var_like(t)) and _unquote_functor(t.functor) == "." and t.arity == 2

def _is_arith_functor(functor: str, arity: int) -> bool:
    # _arithmetic_functions stores keys as (unquote(func), arity)
    return (functor.strip("'"), arity) in _arithmetic_functions

def _const_kind(v: Any) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "bool"
    if isinstance(v, (int, float)):
        return "number"
    if isinstance(v, str):
        return "string"
    return "text"

def _to_loc(program: PrologString, node: Any) -> Optional[SrcLoc]:
    loc = getattr(node, "location", None)
    if loc is None:
        return None
    info = program.lineno(loc, force_filename=True)
    if info is None:
        return None
    filename, line, col = info
    return SrcLoc(file=filename, line=line, col=col)


# ------------------------------------------------------------
# Builtins: retrieve from engine (preferred) or build a temp engine
# ------------------------------------------------------------

# --- replace _collect_builtins() with this version ---

def _collect_builtins() -> Dict[Tuple[str, int], str]:
    """
    Build (functor, arity) -> "functor/arity" map WITHOUT instantiating ClauseDBEngine.
    This is version-robust and avoids abstract engine issues.
    """
    class _BuiltinCollector:
        def __init__(self):
            self.seen: Dict[Tuple[str, int], str] = {}

        def add_builtin(self, name: str, arity: int, fn: Callable):
            functor = _unquote_functor(name)
            a = int(arity)
            self.seen[(functor, a)] = _sig(functor, a)

    eng = _BuiltinCollector()

    # wrappers are required by add_standard_builtins signature; identity is fine here.
    ident = lambda f: f

    # standard engine builtins
    engine_builtin.add_standard_builtins(eng, b=ident, s=ident, sp=ident)

    # registry-decorated builtins (builtin.add_builtins is called inside add_standard_builtins already)
    # but if some versions don't include them there, this makes it explicit and safe:
    try:
        engine_builtin.builtin.add_builtins(eng, ident, ident, ident)
    except Exception:
        pass

    return eng.seen


# ------------------------------------------------------------
# Normalizer core
# ------------------------------------------------------------

@dataclass
class NormCtx:
    base_dir: Optional[Path]       # current source directory (for relative includes)

    visited_files: Set[str] = field(default_factory=set)  # set of already visited file paths
    current_file: Optional[str] = "main.pl"
    current_program: Optional[PrologString] = None

    _program: Optional[ProgramIR] = None  # current output ProgramIR being built
    _irtmap: Optional[IRTMap] = None  # current IRTMap being built

class ProbLogNormalizer:
    """
    Normalize ProbLog AST into unified ProgramIR structure.    
    """
    def __init__(self, 
                 prolog_parser:Optional[PrologString]=None,
                 buildin_map: Optional[Dict[Tuple[str, int], str]] = None, # default: collect from engine
    ) -> None:
        self.builtin_map = buildin_map if buildin_map is not None else _collect_builtins()
        self.prolog_parser = prolog_parser if prolog_parser is not None else PrologString

        self.base_dir = None # It's only for information, not paticipating in normalization logic

        self.norm_ctx: Optional[NormCtx] = None
        self.path_solver = PathSolver()

    def reset(self, ctx:Optional[NormCtx]=None, program:Optional[PrologString]=None, base_dir:Optional[Path]=None) -> None:
        self.norm_ctx = ctx
        self.base_dir = base_dir

    # ---------- Expr conversion ----------
    def to_expr(self, node: Any, *, in_body: bool) -> Expr:
        """
        Convert ProbLog AST node into unified Expr.
        in_body decides whether to treat true/fail and control structures specially.
        """
        # empty body => true
        if node is None and in_body:
            return TrueExpr()

        # Var-like
        if _is_var_like(node):
            name = _var_name(node)
            return VariableExpr(name=name)

        # Constant
        if isinstance(node, Constant):
            v = getattr(node, "value", None)
            # ProbLog "string" constant might be stored with quotes by factory; keep as-is.
            k = _const_kind(v)
            # return ConstantExpr(value=v, kind=k, text=str(node))
            return ConstantExpr(value=v, kind=k)

        # Not class
        if isinstance(node, Not):
            child = getattr(node, "child", None)
            inner = self.to_expr(child, in_body=True)
            # return LogicExpr(op="not", values=[inner], text=str(node))
            return LogicExpr(op="not", values=[inner])

        # And/Or classes
        if isinstance(node, And):
            vals = [self.to_expr(x, in_body=True) for x in self._flatten_and(node)]
            # return LogicExpr(op="and", values=vals, text=str(node))
            return LogicExpr(op="and", values=vals)

        if isinstance(node, Or):
            vals = [self.to_expr(x, in_body=True) for x in self._flatten_or(node)]
            # return LogicExpr(op="or", values=vals, text=str(node))
            return LogicExpr(op="or", values=vals)

        # Term
        if isinstance(node, Term):

            if getattr(node, "_hash_id", None) is not None:
                """
                ####### Special Agent Term handling: #######
                """
                return self._handle_agent_term(node, in_body)

            functor = _unquote_functor(node.functor)
            arity = node.arity
            args = list(getattr(node, "args", ()))

            # Body-specific specials
            if in_body:
                # true/fail
                if functor == "true" and arity == 0:
                    return TrueExpr()
                if functor == "fail" and arity == 0:
                    return FailExpr()

                # \+ as term form (sometimes appears)
                if functor == r"\+" and arity == 1:
                    inner = self.to_expr(args[0], in_body=True)
                    # return LogicExpr(op="not", values=[inner], text=str(node))
                    return LogicExpr(op="not", values=[inner])

                # IfThenElse: (Cond -> Then ; Else)
                # IMPORTANT: match by normalized functors, not raw "'->'"
                if functor == ";" and arity == 2:
                    left, right = args[0], args[1]
                    if isinstance(left, Term) and _unquote_functor(left.functor) == "->" and left.arity == 2:
                        return IfThenElseExpr(
                            condition=self.to_expr(left.args[0], in_body=True),
                            then=self.to_expr(left.args[1], in_body=True),
                            else_=self.to_expr(right, in_body=True),
                            # text=str(node),
                        )

                # IfThen: (Cond -> Then)
                if functor == "->" and arity == 2:
                    return IfThenExpr(
                        condition=self.to_expr(args[0], in_body=True),
                        then=self.to_expr(args[1], in_body=True),
                        # text=str(node),
                    )

                # Flatten comma/semicolon term form if it appears (rare if And/Or classes used)
                if functor == "," and arity == 2:
                    vals = [self.to_expr(x, in_body=True) for x in self._flatten_sep(node, ",")]
                    # return LogicExpr(op="and", values=vals, text=str(node))
                    return LogicExpr(op="and", values=vals)
                if functor == ";" and arity == 2:
                    vals = [self.to_expr(x, in_body=True) for x in self._flatten_sep(node, ";")]
                    # return LogicExpr(op="or", values=vals, text=str(node))
                    return LogicExpr(op="or", values=vals)

                # Builtin?
                if (functor, arity) in self.builtin_map:
                    sig = self.builtin_map[(functor, arity)]
                    b = BuiltinExpr(
                        functor=functor,
                        arguments=[self.to_expr(a, in_body=True) for a in args],
                        signature=sig,
                        category=self._builtin_category(functor, arity),
                        # text=str(node),
                        defines=[],
                        uses=[],
                    )
                    # Conservative defines/uses:
                    # - is/2 defines LHS var, uses RHS vars
                    # - other builtins: treat all vars as uses
                    defines, uses = self._builtin_flow(functor, args)
                    b = BuiltinExpr(
                        functor=b.functor,
                        arguments=b.arguments,
                        signature=b.signature,
                        category=b.category,
                        # text=b.text,
                        defines=defines,
                        uses=sorted(uses),
                    )
                    return b

            # Lists
            if _is_nil(node):
                # return ListExpr(elements=[], tail=None, text=str(node))
                return ListExpr(elements=[], tail=None)
            if _is_list_cell(node):
                elems: List[Expr] = []
                cur = node
                while _is_list_cell(cur):
                    elems.append(self.to_expr(cur.args[0], in_body=False))
                    cur = cur.args[1]
                tail_expr = None if _is_nil(cur) else self.to_expr(cur, in_body=False)
                # return ListExpr(elements=elems, tail=tail_expr, text=str(node))
                return ListExpr(elements=elems, tail=tail_expr)

            # Arithmetic function, including pi/0
            if _is_arith_functor(functor, arity):
                return ArithmeticExpr(
                    functor=functor,
                    arguments=[self.to_expr(a, in_body=False) for a in args],
                    signature=_sig(functor, arity),
                    # text=str(node),
                )

            # Atom (0-arity) -> TermExpr but signature has arity 0
            if arity == 0:
                if in_body:
                    return TermExpr(functor=functor, arguments=None, signature=_sig(functor, 0))
                return AtomExpr(name=functor)

            # Normal compound term
            return TermExpr(
                functor=functor,
                arguments=[self.to_expr(a, in_body=False) for a in args],
                signature=_sig(functor, arity),
                # text=str(node),
            )

        # Fallback: degrade to Const(text)
        # return ConstantExpr(value=str(node), kind="text", text=str(node))
        return ConstantExpr(value=str(node), kind="text")


    # ===================== agent statement handling ===================== # 
    # Normalize single head for agent statement, we use a trick here which is use head+"." to parse easily
    def normalize_single_head(self, head: str) -> TermExpr | AtomExpr:
        """Normalize a single head string into TermExpr or AtomExpr."""
        head_term = PrologString(head + ".", parser=self.prolog_parser)
        head_args = list(head_term)

        assert len(head_args) == 1, f"Expected single statement in head: {head}"

        st = head_term[0]
        assert isinstance(st, Term), f"Expected Term in head: {type(st)}"

        head_expr = self.to_expr(st, in_body=False)
        if isinstance(head_expr, AtomExpr):
            head_expr = TermExpr(functor=head_expr.name, arguments=None, signature=_sig(head_expr.name, 0))
        assert isinstance(head_expr, TermExpr)
        return head_expr

    # Agent Term handler
    def _handle_agent_term(self, node) -> None:
        """
        Special handling for Agent Term: __AGENT_@_TERM__(IRTDict)
        Notice:
        1. It will register the AgentStmt DIRECTLY into norm_ctx._program.statements
        2. It will return AgentMarkExpr for AST tree replacement.
        """
        # register into IRTMap:
        irt_dict:IRTDict = node.args[0]
        self.norm_ctx._irtmap.add_dict(irt_dict)

        # assert isinstance(irt_dict, IRTDict), f"Expected IRTDict in Agent Term, got: {type(irt_dict)}"
        # AgentStmt:
        if irt_dict.initials.has_unknown_args:
            head = TermExpr()
        else:
            head_ctx = irt_dict.get_head_ctx()
            head = self.normalize_single_head(head_ctx)

        # register AgentStmt to program statements directly:
        agent_mark_expr = AgentMarkExpr(
            id=irt_dict.initials.hash_id,
            functor=head.functor,
            arguments=head.arguments, # can be None
            signature=head.signature, # can be None
            partial=irt_dict.initials.has_unknown_args,
        )
        self.norm_ctx._program.statements.append(
            AgentStmt(
                id=irt_dict.initials.hash_id,
                head=agent_mark_expr,
                name=irt_dict.initials.agent_name,
                prob=float(getattr(node, "probability", None)) if hasattr(node, "probability") else None,
                text=irt_dict.builtins.Agent,
                loc=_to_loc(self.norm_ctx.current_program, node),
            )
        )
        # return AgentMarkExpr for the ast tree:
        return agent_mark_expr

    # ---------- flatten helpers ----------
    def _flatten_and(self, node: Any) -> List[Any]:
        out: List[Any] = []
        def rec(x: Any):
            if isinstance(x, And):
                rec(x.op1); rec(x.op2)
            else:
                out.append(x)
        rec(node)
        return out

    def _flatten_or(self, node: Any) -> List[Any]:
        out: List[Any] = []
        def rec(x: Any):
            if isinstance(x, Or):
                rec(x.op1); rec(x.op2)
            else:
                out.append(x)
        rec(node)
        return out

    def _flatten_sep(self, term: Term, sep: str) -> List[Any]:
        out: List[Any] = []
        def rec(x: Any):
            if isinstance(x, Term) and _unquote_functor(x.functor) == sep and x.arity == 2:
                rec(x.args[0]); rec(x.args[1])
            else:
                out.append(x)
        rec(term)
        return out

    # ---------- builtin analysis ----------
    def _builtin_category(self, functor: str, arity: int) -> str:
        # Minimal coarse categorization (optional; safe defaults)
        if functor in {"is"} and arity == 2:
            return "arithmetic"
        if functor in {">", "<", ">=", "=<", "=:=", "=\\="} and arity == 2:
            return "compare"
        if functor in {"=", "\\="} and arity == 2:
            return "unify"
        if functor in {"call", "once", "findall", "all", "subquery"}:
            return "meta"
        if functor in {"write", "writeln", "writenl", "nl", "debugprint", "error"}:
            return "io"
        return "other"

    def _vars_in_expr(self, e: Expr, out: Set[str]) -> None:
        # only VariableExpr appear after normalization
        if isinstance(e, VariableExpr):
            if e.name and e.name != "_":
                out.add(e.name)
            return
        if isinstance(e, LogicExpr):
            for v in e.values:
                self._vars_in_expr(v, out)
            return
        if isinstance(e, (IfThenExpr, IfThenElseExpr)):
            self._vars_in_expr(e.condition, out)
            self._vars_in_expr(e.then, out)
            if isinstance(e, IfThenElseExpr):
                self._vars_in_expr(e.else_, out)
            return
        if isinstance(e, BuiltinExpr):
            for a in e.arguments:
                self._vars_in_expr(a, out)
            return
        if isinstance(e, TermExpr):
            for a in e.arguments:
                self._vars_in_expr(a, out)
            return
        if isinstance(e, ListExpr):
            for el in e.elements:
                self._vars_in_expr(el, out)
            if e.tail is not None:
                self._vars_in_expr(e.tail, out)
            return

    def _builtin_flow(self, functor: str, raw_args: List[Any]) -> Tuple[List[str], Set[str]]:
        """
        Conservative flow on raw ProbLog args.
        We compute defines/uses AFTER conversion.
        """
        defines: List[str] = []
        uses: Set[str] = set()

        # Convert args to Expr (body context)
        args_expr = [self.to_expr(a, in_body=True) for a in raw_args]

        if functor == "is" and len(args_expr) == 2:
            lhs, rhs = args_expr[0], args_expr[1]
            if isinstance(lhs, VariableExpr) and lhs.name != "_":
                defines.append(lhs.name)
            self._vars_in_expr(rhs, uses)
            return defines, uses

        # other builtins: all vars as uses
        for a in args_expr:
            self._vars_in_expr(a, uses)
        return defines, uses

    # ---------- main normalization ----------
    def normalize(
            self, 
            code: str,
            *,
            path: Optional[str] = None, # The upper process should process to get base dir!!!
            is_agent: bool = False, # For later agent-specific processing
    ) -> Tuple[ProgramIR, IRTMap]:
        """Normalize a full ProbLog program string into ProgramIR."""
        self.base_dir = self.path_solver.get_dir()
        self.norm_ctx = NormCtx(
            base_dir=self.base_dir,
            visited_files=set(),
            current_file=self.path_solver.get_filename(path) if path is not None else "main.pl",

            _program = ProgramIR(from_agent=is_agent, source_text=code),
            _irtmap = IRTMap()
        )

        try:
            out_program = self._normalize_program(
                            code,
                            is_agent=is_agent,
                            source_file=None,
                            base_dir=self.base_dir,
                        )
            out_irtmap = self.norm_ctx._irtmap
            return out_program, out_irtmap
        finally:
            self.reset()

    def _normalize_program(
            self, 
            code:str,
            is_agent:bool,
            source_file:Optional[str]=None,
            base_dir:Optional[str]=None, # 
            # Input additional python text as string for extern spec parsing 
    ) -> ProgramIR:
        """Normalize a full ProbLog program string into ProgramIR."""
        self.norm_ctx.current_program = PrologString(code, parser=self.prolog_parser)
        # establish base_dir for this parsing context
        prev_base_dir = self.norm_ctx.base_dir
        if base_dir is not None:
            self.norm_ctx.base_dir = self.path_solver.get_dir(base_dir)
        elif source_file is not None:
            self.norm_ctx.base_dir = self.path_solver.get_base_dir(source_file)
        else:
            # keep previous if already inside a file include; otherwise None
            self.norm_ctx.base_dir = prev_base_dir

        try:
            for st in self.norm_ctx.current_program:
                out = self._normalize_statement(self.norm_ctx.current_program, st, is_agent)
                if isinstance(out, list):
                    self.norm_ctx._program.statements.extend(out)
                elif out is not None:
                    self.norm_ctx._program.statements.append(out)
        finally:
            # restore base_dir (important for nested recursion)
            self.norm_ctx.base_dir = prev_base_dir

        return self.norm_ctx._program

    def _normalize_statement(self, 
                             program: PrologString,
                             st:StatementBase, 
                             is_agent:bool
        ) -> Optional[StatementBase | List[StatementBase]]:
        """Convert a single ProbLog statement into unified StatementBase."""

        # Fact: Term (not Clause/AD)
        if isinstance(st, Term) and not isinstance(st, (Clause, AnnotatedDisjunction)):

            if st.functor == "__AGENT_@_TERM__":
                """
                Special Agent Term handling:
                """
                return self._handle_agent_term(st, in_body=False)

            head = self.to_expr(st, in_body=False)
            assert isinstance(head, (TermExpr, AtomExpr)), f"fact head unexpected: {type(head)}"

            # 如果你坚持保留 AtomExpr（0-arity），这里统一转成 TermExpr，方便下游
            if isinstance(head, AtomExpr):
                head = TermExpr(functor=head.name, arguments=None, signature=_sig(head.name, 0))

            # 一定要在这里 return，不要放在上面那个 if 里
            return FactStmt(
                head=head,
                prob=float(getattr(st, "probability", None) or 1.0),
                text=str(st),
                loc=_to_loc(program, st),
            )

        # Rule: Clause
        if isinstance(st, Clause):
            head_term = st.head
            body_term = st.body

            head = self.to_expr(head_term, in_body=False)
            if isinstance(head, AtomExpr):
                head = TermExpr(functor=head.name, arguments=None, signature=_sig(head.name, 0))
            assert isinstance(head, TermExpr)

            body_expr = self.to_expr(body_term, in_body=True)

            # include external files as ImportStmts: 
            #   ":- include('file.pl')."" or ":- include('file.py')."
            # directive: no head, body is include("x")
            is_directive = _unquote_functor(head.functor) == "_directive" and head_term.arity == 0
            if is_directive and isinstance(body_term, Term) and _unquote_functor(body_term.functor) == "include" and body_term.arity == 1:

                # include directive
                prev_file_name = self.norm_ctx.current_file
                file_name = _unquote_functor(str(body_term.args[0]))
                print(f"Including file: {file_name}")
                if not file_name:
                    raise ValueError(f"Empty file name in include directive at {self.norm_ctx.current_program.lineno(getattr(st, 'location', None))}")
                
                self.norm_ctx.current_file = file_name
                path = self.path_solver.get_path(file_name, give_base_dir=self.norm_ctx.base_dir)
                abs_path_str = str(path)

                suffix = path.suffix.lower()
                if abs_path_str in self.norm_ctx.visited_files:
                    # already included; skip
                    return None
                self.norm_ctx.visited_files.add(abs_path_str)

                file_content = path.read_text(encoding="utf-8")
                if suffix == ".py": # python module
                    file_type = "python"

                    # ========?????? Python file handling ?????========
                    file_content = textwrap.dedent(file_content)
                    extern_info = extern_arities(file_content) if file_content else {}

                    extern_stmts: List[ExternalStmt] = []
                    for func_name, meta in extern_info.items():
                        # Idk wether arguments are needed here...
                        head = TermExpr(functor=func_name, arguments=None, signature=_sig(func_name, len(meta.get("mode_spec", []))))
                        extern_stmt = ExternalStmt(
                            head=head,
                            mode_spec=meta.get("mode_spec", []),
                            docstring=meta.get("docstring", ""),
                        )
                        extern_stmts.append(extern_stmt)

                    self.norm_ctx._program.statements.extend(extern_stmts)

                elif suffix in {".pl", ".problog", ".pblog", ".lp"}: # problog file
                    file_type = "problog"
                    # Run normalization recursively to expand included files!!!
                    self._normalize_program(
                        file_content,
                        is_agent=is_agent,
                        source_file=abs_path_str,
                        base_dir=path.parent,
                    )
                else:
                    file_type = "other" # unknown type, just import as-is
                
                try:
                    return ImportStmts(
                        file_type=file_type,
                        file_path=abs_path_str,
                        text=str(st),
                        loc=_to_loc(program, st),
                    )
                finally:
                    self.norm_ctx.current_file = prev_file_name

            return RuleStmt(
                head=head,
                body=body_expr,
                prob=float(getattr(head_term, "probability", None) or 1.0),
                text=str(st),
                loc=_to_loc(program, st),
            )

        # AD: AnnotatedDisjunction
        if isinstance(st, AnnotatedDisjunction):
            body_expr = self.to_expr(st.body, in_body=True)

            rules = []
            for h in st.heads:
                hx = self.to_expr(h, in_body=False)
                if isinstance(hx, AtomExpr):
                    hx = TermExpr(functor=hx.name, arguments=None, signature=_sig(hx.name, 0))
                assert isinstance(hx, TermExpr)

                prob = float(getattr(h, "probability", None) or 1.0)
                try:
                    str(st)
                except Exception:
                    raise ValueError(f"Please do not define Agent Term inside AD Clauses: '{h}'")

                rules.append(RuleStmt(
                    head=hx,
                    body=body_expr,
                    prob=prob,
                    text=f"{h} :- {st.body}",
                    loc=_to_loc(program, st),
                ))
            return rules
            
        return StatementBase(type="Unknown", text=str(st), loc=_to_loc(program, st))


