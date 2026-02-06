# ast_data_structure_unified.py
# A unified IR schema designed to work directly with the “baseline code”
# (ProbLog official parser + engine.get_builtins() classification).
from __future__ import annotations

import dataclasses, inspect, json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Set, Union, Tuple
from models.agent_structure import HashID

def to_jsonable(x: Any) -> Any:
    # dataclass -> dict
    if dataclasses.is_dataclass(x):
        d = {k: to_jsonable(v) for k, v in dataclasses.asdict(x).items() if v is not None}
        for name, _ in inspect.getmembers(type(x), lambda o: isinstance(o, property)):
            if name.startswith("_") or name in d:
                continue
            try:
                v = getattr(x, name)
            except Exception:
                continue
            if v is not None:
                d[name] = to_jsonable(v)
        return d

    # containers
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items() if v is not None}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x if v is not None]
    if isinstance(x, set):
        # keep stable order for readability
        return sorted(to_jsonable(v) for v in x if v is not None)
    if isinstance(x, property):
        return x

    # primitives
    if x is None or isinstance(x, (str, int, float, bool)):
        return x

    # fallback
    return str(x)


# ============================================================
# 1) Core Expr
#    - Covers: Var / Const / Atom / Term / List / Logic / Builtin / Control
#    - Designed to be:
#        (a) close to ProbLog official AST
#        (b) safe for dependency extraction / prompt usage
# ============================================================

@dataclass(frozen=True)
class VariableExpr:
    type: Literal["Var"] = "Var"
    name: str = ""                       # normalized: X / A1 / X3 / _


@dataclass(frozen=True)
class ConstantExpr:
    """
    Literal constants coming from problog.logic.Constant:
      - number / string / bool / null
    Also supports a degraded "text" constant if needed.
    """
    type: Literal["Const"] = "Const"
    value: Any = None
    kind: Optional[Literal["number", "string", "bool", "null", "text"]] = None
    # text: Optional[str] = None           # original str(...) fallback


@dataclass(frozen=True)
class AtomExpr:
    """
    0-arity atom (a, red, foo).
    Keep it separate from TermExpr to avoid polluting "predicate deps".
    """
    type: Literal["Atom"] = "Atom"
    name: str = ""
    # text: Optional[str] = None

@dataclass(frozen=True)
class TermExpr:
    """
    General compound term / predicate call (arity >= 1) or explicitly modeled term.
    NOTE: prefer AtomExpr for 0-arity atoms in most cases.
    """
    functor: str = ""
    type: Literal["Term"] = "Term"
    arguments: Optional[List["Expr"]] = None  # None for 0-arity terms

    # Optional but highly recommended: exact signature for deps (functor/arity)
    signature: Optional[str] = None      # e.g. "temp/2"
    # text: Optional[str] = None


@dataclass(frozen=True)
class ArithmeticExpr:
    """
    Arithmetic expression (e.g. 3 + X * 2).
    Stored as a TermExpr with known arithmetic functor.
    """
    type: Literal["Arithmetic"] = "Arithmetic"
    functor: str = ""
    arguments: List["Expr"] = field(default_factory=list)
    signature: Optional[str] = None      # e.g. "temp/2"
    # text: Optional[str] = None


@dataclass(frozen=True)
class ListExpr:
    """
    Prolog list:
      []      => elements=[], tail=None
      [a,b]   => elements=[a,b], tail=None
      [H|T]   => elements=[H],   tail=T
      [a,b|T] => elements=[a,b], tail=T
    """
    type: Literal["List"] = "List"
    elements: List["Expr"] = field(default_factory=list)
    tail: Optional["Expr"] = None
    # text: Optional[str] = None


@dataclass(frozen=True)
class LogicExpr:
    """
    Flattened logical structure for bodies:
      And:  A,B,C  => op="and", values=[A,B,C]
      Or:   A;B;C  => op="or",  values=[A,B,C]
      Not:  \\+A    => op="not", values=[A]
    """
    type: Literal["Logic"] = "Logic"
    op: Literal["and", "or", "not"] = "and"
    values: List["Expr"] = field(default_factory=list)
    # text: Optional[str] = None


@dataclass(frozen=True)
class IfThenExpr:
    """
    Control structure: (Cond -> Then)
    """
    type: Literal["IfThen"] = "IfThen"
    condition: "Expr" = None
    then: "Expr" = None
    # text: Optional[str] = None


@dataclass(frozen=True)
class IfThenElseExpr:
    """
    Control structure: (Cond -> Then ; Else)
    """
    type: Literal["IfThenElse"] = "IfThenElse"
    condition: "Expr" = None
    then: "Expr" = None
    else_: "Expr" = None
    # text: Optional[str] = None


@dataclass(frozen=True)
class BuiltinExpr:
    """
    Builtin call (decided by engine.get_builtins()).
    Store both exact signature and an optional coarse category.
    """
    type: Literal["Builtin"] = "Builtin"
    functor: str = ""                          # e.g. "is", "=", ">", "\\+"
    arguments: List["Expr"] = field(default_factory=list)

    signature: Optional[str] = None            # e.g. "is/2", ">/2", "findall/3"
    category: Optional[str] = None             # e.g. "arithmetic|compare|unify|meta|io|control|other"

    # Optional: conservative dataflow
    defines: List[str] = field(default_factory=list)
    uses: List[str] = field(default_factory=list)

    # text: Optional[str] = None


@dataclass(frozen=True)
class TrueExpr:
    type: Literal["True"] = "True"
    # text: Optional[str] = "true"


@dataclass(frozen=True)
class FailExpr:
    type: Literal["Fail"] = "Fail"
    # text: Optional[str] = "fail"


@dataclass(frozen=True)
class AgentMarkExpr:
    """
    It's only used to mark where the agent is called, keep the ast structure.
    If it has signature, then functor and arguments are filled accordingly.
    Else then, just use id as the unique identity.
    """
    id:HashID
    type: Literal["AgentMark"] = "AgentMark"
    functor: str = ""
    arguments: Optional[List["Expr"]] = None  # None for 0-arity terms

    signature: Optional[str] = None      # e.g. "head/1"
    partial: bool = False # whether has unknown args
    # text: Optional[str] = None


Expr = Union[
    VariableExpr,
    ConstantExpr,
    AtomExpr,
    TermExpr,
    ArithmeticExpr,
    ListExpr,
    LogicExpr,
    IfThenExpr,
    IfThenElseExpr,
    BuiltinExpr,
    TrueExpr,
    FailExpr,
    AgentMarkExpr,
]


# ============================================================
# 3) Statements (top-level)
#    - Fact / Rule / Annotated Disjunction
#    - Keep "text" and "loc" for debugging & traceability
# ============================================================

@dataclass(frozen=True)
class SrcLoc:
    """
    Maps to LogicProgram.lineno(node.location):
      (filename, line, col) or (None, line, col)
    """
    file: Optional[str] = None
    line: Optional[int] = None
    col: Optional[int] = None


@dataclass
class StatementBase:
    type: str
    text: Optional[str] = None
    loc: Optional[SrcLoc] = None
    # analysis: BodyAnalysis = field(default_factory=BodyAnalysis)  # NEW
    def to_dict(self) -> dict:
        return to_jsonable(self)

@dataclass
class FactStmt(StatementBase):
    type: Literal["Fact"] = "Fact"
    head: TermExpr = None
    prob: float = 1.0                        # extracted probability (default 1.0)


@dataclass
class RuleStmt(StatementBase):
    type: Literal["Rule"] = "Rule"
    head: TermExpr = None
    body: Expr = None
    prob: float = 1.0                        # clause weight if you store it (often 1.0)
    # analysis: BodyAnalysis = field(default_factory=BodyAnalysis)


@dataclass
class ImportStmts(StatementBase):
    type: Literal["Import"] = "Import"
    file_path: Optional[str] = None       # resolved absolute path if available
    file_type: Literal["problog", "python"] = "problog"  # 'file' for include/import, 'module' for python module

@dataclass
class ExternalStmt(StatementBase):
    type: Literal["External"] = "External"
    head: TermExpr = None
    mode_spec: List[str] = field(default_factory=list)
    docstring: str = ""

@dataclass
class AgentStmt(StatementBase):
    """After construction it should contains all information of initials:
    'head' contains functor=>head_name; arguments=>head_args;partial=>has_unknown_args
    'name' => agent_name    
    'text' => builtins.Agent"""
    id:HashID = None
    type: Literal["Agent"] = "Agent"
    head: AgentMarkExpr = None # head term
    name: str = None # Agent name
    prob: Optional[float] = None
    # instructions => stored in 'text' field

# @dataclass
# class ADHead:
#     head: TermExpr
#     prob: float = 1.0
#     text: str = ""


# @dataclass
# class ADStmt(StatementBase):
#     type: Literal["AD"] = "AD"
#     heads: List[ADHead] = field(default_factory=list)
#     body: Expr = None
#     # analysis: BodyAnalysis = field(default_factory=BodyAnalysis)

Statement = Union[FactStmt, RuleStmt, ImportStmts]

@dataclass
class ProgramIR:
    """The full program IR. This also useful for later real analysis passes."""
    id: Optional[str] = None # if from_agent, It's the hash id of the agent
    from_agent: bool = False
    statements: List[StatementBase] = field(default_factory=list)
    source_text: Optional[str] = None
    def to_dict(self) -> dict:
        return to_jsonable(self)

# ============================================================
# 4) Grouping layer (your existing “FactGroup / RuleGroup / ExternalSpec”)
#    - Keep as-is, but align field names with the unified schema.
# ============================================================

@dataclass
class ProbabilitySummary:
    minimum: float
    maximum: float
    varies: bool

@dataclass
class Dependencies:
    predicate_deps: Set[str] = field(default_factory=set)
    builtin_deps: Set[str] = field(default_factory=set)

@dataclass
class PredicateGroup:
    id: str
    signature: str # if signature is hash id, it can only contains one single agent

    # raw items
    fact_count: int = 0
    fact_samples: List[FactStmt] = field(default_factory=list)  # samples only; not necessarily all facts
    rule_count: int = 0
    rules: List[RuleStmt] = field(default_factory=list)
    # extern (optional)
    external_spec: Optional[ExternalStmt] = None # None if not external
    # agent:
    agents: Optional[List[AgentStmt]] = None

    # stats
    # If there's more than one fact/rules, use probability_summary, otherwise use probability
    # probability: Optional[float] = None  # average prob for facts only
    # probabilities_summary: Optional[ProbabilitySummary] = None

    # deps: only meaningful when rules exist, but keep empty sets to avoid None-branches
    dependencies: Dependencies = field(default_factory=Dependencies)
    agentic_dependencies: Optional[Dependencies] = None  # deps introduced by agent calls
    # LLM-filled
    summary: str = ""
    io_examples: Optional[List[Tuple[str, str]]] = None  # set of (input, output) examples

    @property
    def group_kind(self) -> Literal["facts_only", "rules_only", "external_only", 
                            "non_agentic_mixed", "agentic_mixed", "pure_agentic"]:
        has_facts, has_rules, has_external  = self.fact_count > 0, self.rule_count > 0, self.external_spec is not None
        if self.agents is not None:
            if not has_facts and not has_rules and not has_external: return "pure_agentic"
            else: return "agentic_mixed"
        if has_facts and not has_rules and not has_external: return "facts_only"
        elif not has_facts and has_rules and not has_external: return "rules_only"
        elif not has_facts and not has_rules and has_external: return "external_only"
        else: return "non_agentic_mixed"

    def to_dict(self) -> dict:
        return to_jsonable(self)


# -----------------------------
# Unified group model
# -----------------------------
def dump_ir(ir) -> str:
    return json.dumps(ir, ensure_ascii=False, indent=2)

@dataclass
class PredicateGroupsIR:
    version: str = "main.pl-groups-ir"
    defined_signatures: Set[str] = field(default_factory=set)
    used_signatures: Set[str] = field(default_factory=set)

    @property
    def all_signatures(self) -> Set[str]:
        return self.defined_signatures | self.used_signatures

    used_builtins: Set[str] = field(default_factory=set)
    imported_files: Set[str] = field(default_factory=set)

    # Groups: signature -> group
    groups: Dict[str, "PredicateGroup"] = field(default_factory=dict)  

    # -------------------------
    # basic helpers (internal)
    # -------------------------
    def to_dict(self) -> dict:
        return to_jsonable(self)

    def get_all_signatures(self) -> List[str]:
        # stable order is usually better for caching/diff/prompt
        return sorted(list(self.all_signatures))

    def get_group(self, signature: str) -> Optional["PredicateGroup"]:
        return self.groups.get(signature, None)

    def get_dependencies(self, include_empty: bool = False) -> Dict[str, List[str]]:
        """
        Return a dependency graph: {predicate_signature: [dependent predicate_signatures]}.
        - include_empty=False: omit nodes with empty deps.
        - include_empty=True: include all nodes (deps may be []).
        """
        dep_graph: Dict[str, List[str]] = {}
        for sig in sorted(self.groups.keys()):
            group = self.groups[sig]
            deps_list = sorted(list(group.dependencies.predicate_deps)) if group.dependencies else []
            if (not deps_list) and (not include_empty):
                continue
            dep_graph[sig] = deps_list
        return dep_graph


    def update_agent_head(self, key: HashID, new_head: TermExpr) -> None:
        """Update the head of all AgentStmt in the group to new_head.
        Which group are required to update head? =>  the group that only contains one single agent statement, 
        because if any of the Statement has head, or has more than one statements, means they already
        have signature, because 'signature' is required for grouping.
        """
        group = self.get_group(key)
        if group is None:
            raise ValueError(f"No group found for key {key} to update agent head.")
        assert group.has_agent and len(group.agents) == 1, f"Group {group.id} must have exactly one AgentStmt to update head."
        group.signature = new_head.signature # also update the group signature
        agent_stmt = group.agents[0]
        agent_stmt.head = new_head
   


    # ============================================================
    # Views (LLM-facing)
    # ============================================================

    # --------
    # helpers
    # --------
    def _group_to_min_dict(
        self,
        group: "PredicateGroup",
        *,
        max_fact_samples: int = 2,
        max_rules: Optional[int] = None,
        include_external_doc: bool = True,
        include_summary: bool = False,
        include_io_examples: bool = False,
        include_prob: bool = True,
        include_deps: bool = True,
        include_loc: bool = False,
        include_body: bool = False,
    ) -> dict:
        d: Dict[str, Any] = {
            "id": group.id,
            "signature": group.signature,
            "group_kind": group.group_kind,
            "fact_count": group.fact_count,
            "rule_count": group.rule_count,
            "has_external": group.has_external,
        }

        # facts (samples)
        if group.fact_count > 0 and group.fact_samples is not None and max_fact_samples != 0:
            samples = group.fact_samples
            if max_fact_samples is not None and max_fact_samples > 0:
                samples = samples[:max_fact_samples]
            if include_loc:
                d["fact_samples"] = [{"text": s.text, "loc": to_jsonable(s.loc)} for s in samples]
            else:
                d["fact_samples"] = [s.text for s in samples]

        # rules (text + optional body/loc)
        if group.rule_count > 0 and group.rules is not None and max_rules != 0:
            rules = group.rules
            if max_rules is not None and max_rules > 0:
                rules = rules[:max_rules]

            if include_body or include_loc:
                out_rules = []
                for r in rules:
                    rr: Dict[str, Any] = {"text": r.text}
                    if include_loc:
                        rr["loc"] = to_jsonable(r.loc)
                    if include_body:
                        rr["head"] = to_jsonable(r.head)
                        rr["body"] = to_jsonable(r.body)
                        rr["prob"] = r.prob
                    out_rules.append(rr)
                d["rules"] = out_rules
            else:
                d["rules"] = [r.text for r in rules]

        # external
        if group.has_external and group.external_spec is not None:
            ext: Dict[str, Any] = {"mode_spec": list(group.external_spec.mode_spec)}
            if include_external_doc:
                ext["docstring"] = group.external_spec.docstring
            d["external_spec"] = ext

        # probability info
        if include_prob:
            if group.probability is not None:
                d["probability"] = group.probability
            if group.probabilities_summary is not None:
                d["probabilities_summary"] = dataclasses.asdict(group.probabilities_summary)

        # deps
        if include_deps and group.dependencies is not None:
            d["dependencies"] = {
                "predicate_deps": sorted(list(group.dependencies.predicate_deps)),
                "builtin_deps": sorted(list(group.dependencies.builtin_deps)),
            }

        # optional LLM fields
        if include_summary:
            d["summary"] = group.summary
        if include_io_examples:
            d["io_examples"] = list(group.io_examples)

        return d

    def _resolve_dependency_closure(self, target_sig: str) -> List[str]:
        """
        Return dependency closure (predicate deps) including target_sig.
        Order: deps-first, target-last (stable).
        """
        if target_sig not in self.groups:
            return [target_sig]

        visited: Set[str] = set()
        order: List[str] = []

        def dfs(sig: str):
            if sig in visited:
                return
            visited.add(sig)
            g = self.groups.get(sig)
            if g is not None and g.dependencies is not None:
                for dep in sorted(list(g.dependencies.predicate_deps)):
                    dfs(dep)
            order.append(sig)

        dfs(target_sig)
        return order
