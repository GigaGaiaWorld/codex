from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .ast_structure import (
    # Expr types
    Expr, VariableExpr, ConstantExpr, AtomExpr,
    TermExpr, BuiltinExpr, ArithmeticExpr, ListExpr,
    LogicExpr, IfThenExpr, IfThenElseExpr, TrueExpr, FailExpr, AgentMarkExpr,
    ProgramIR,
    # group helpers
    ProbabilitySummary, PredicateGroup, PredicateGroupsIR, AgentStmt,
)
"""
This module defines the PredicateGroupsIR structure and the function to build groups from ProgramIR.
A "group" is a collection of facts and rules sharing the same predicate signature, along with metadata such as dependencies, summaries, and examples.
For example:
    Predicate: parent/2
    Group:
        Facts:
            parent(john, mary).
            parent(mary, alice).
        Rules:
            parent(X, Y) :- mother(X, Y).
            parent(X, Y) :- father(X, Y).
        Dependencies:
            predicate_deps: {mother/2, father/2}
            builtin_deps: {}
        Summary: "Defines parent-child relationships."
        IO Examples:
            Input: parent(john, X) => Output: {mary}

"""

# -----------------------------
# utilities
# -----------------------------

def _mk_group_id(prefix: str, signature: str) -> str:
    return f"{prefix}_{signature.replace('/', '_')}"

def _prob_summary(ps: List[float]) -> Optional[ProbabilitySummary]:
    if not ps:
        return None
    mn = min(ps)
    mx = max(ps)
    return ProbabilitySummary(minimum=mn, maximum=mx, varies=(abs(mn - mx) > 1e-12))


# -----------------------------
# dep collector
# -----------------------------

class DepCollector:
    def collect(self, expr: Expr | None) -> Tuple[Set[str], Set[str]]:
        pred: Set[str] = set()
        bult: Set[str] = set()
        self._rec(expr, pred, bult)
        return pred, bult

    def _rec(self, e: Expr | None, pred: Set[str], bult: Set[str]) -> None:
        if e is None:
            return

        if isinstance(e, (VariableExpr, ConstantExpr, AtomExpr, TrueExpr, FailExpr)):
            return

        if isinstance(e, BuiltinExpr):
            if getattr(e, "signature", ""):
                bult.add(e.signature)
            for a in getattr(e, "arguments", []) or []:
                self._rec(a, pred, bult)
            return

        if isinstance(e, TermExpr):
            if getattr(e, "signature", ""):
                pred.add(e.signature)
            for a in getattr(e, "arguments", []) or []:
                self._rec(a, pred, bult)
            return

        if isinstance(e, LogicExpr):
            for v in getattr(e, "values", []) or []:
                self._rec(v, pred, bult)
            return

        if isinstance(e, IfThenExpr):
            self._rec(e.condition, pred, bult)
            self._rec(e.then, pred, bult)
            return

        if isinstance(e, IfThenElseExpr):
            self._rec(e.condition, pred, bult)
            self._rec(e.then, pred, bult)
            self._rec(e.else_, pred, bult)
            return

        if isinstance(e, ListExpr):
            for el in getattr(e, "elements", []) or []:
                self._rec(el, pred, bult)
            if getattr(e, "tail", None) is not None:
                self._rec(e.tail, pred, bult)
            return

        if isinstance(e, ArithmeticExpr):
            for a in getattr(e, "arguments", []) or []:
                self._rec(a, pred, bult)
            return



# -----------------------------
# build groups
# -----------------------------

class ProgramGroupsIRBuilder:
    def get_group(self, ir: PredicateGroupsIR, sig: str) -> PredicateGroup:
        g = ir.groups.get(sig)
        if g is None:
            g = PredicateGroup(id=_mk_group_id("pred", sig), signature=sig)
            ir.groups[sig] = g
        return g

    def build_groups(self, 
                     program_ir:ProgramIR, 
                     *, 
                     max_fact_samples:int = 5, 
                     current_groups:Optional[PredicateGroupsIR]=None,
    ) -> PredicateGroupsIR:
        """
        Args:
        - program_ir: ProgramIR
        - extern_source_py: str, optional Python source code to extract extern arities
        - max_fact_samples: int, maximum number of fact samples to keep per group
        Returns:
        - ProgramGroupsIR => the grouped clauses with dependencies and other metadata
        """
        if program_ir.from_agent:
            raise NotImplementedError("Building groups from agent-originated ProgramIR is not supported yet.")

        ir = current_groups or PredicateGroupsIR()
        depc = DepCollector()

        # 1) collect defined + fill groups with facts/rules
        prob_list:Dict[str, List[float]] = {}

        for st in program_ir.statements:
            t = getattr(st, "type", None)
            head = getattr(st, "head", None)
            sig = getattr(head, "signature", "")

            if t == "Agent":
                # =================== AgentStmt =================== #
                # Here, if the agent has a complete head (with signature) => sig is used
                # Otherwise, we use the id as signature
                id = getattr(st, "id") # use id as signature if no signature
                sig = sig or id # use id as signature if no signature
                """From this we create two different cases:
                1) AgentStmt with signature => share head with other facts/rules
                2) AgentStmt without signature => unique group for itself
                However, both cases could update agent content independently."""
                g = self.get_group(ir, sig)
                if sig not in prob_list:
                    prob_list[sig] = []
                g.agents = g.agents or []
                g.agents.append(st)

            elif t == "Import" and getattr(st, "file_type", "") == "python":
                # currently, we only track imported python file names, other attributes are ignored
                ir.imported_files.add(getattr(st, "file_name", ""))

            elif t == "Fact":
                if not sig:
                    continue

                ir.defined_signatures.add(sig)
                g = self.get_group(ir, sig)
                g.fact_count += 1
                if sig not in prob_list:
                    prob_list[sig] = []
                prob_list[sig].append(st.prob)

                # sample only
                if len(g.fact_samples) < max_fact_samples:
                    g.fact_samples.append(st)

            elif t == "Rule":
                if not sig:
                    continue

                ir.defined_signatures.add(sig)
                g = self.get_group(ir, sig)
                g.rule_count += 1
                if sig not in prob_list:
                    prob_list[sig] = []
                prob_list[sig].append(st.prob)

                g.rules.append(st)
                body = getattr(st, "body", None)
                pred_deps, builtin_deps = depc.collect(body)

                if sig in pred_deps:
                    pred_deps.remove(sig)

                g.dependencies.predicate_deps.update(pred_deps)
                g.dependencies.builtin_deps.update(builtin_deps)

                ir.used_signatures.update(pred_deps)
                ir.used_builtins.update(builtin_deps)

            # External statements, if any
            elif t == "External":
                if not sig:
                    continue

                ir.used_signatures.update(sig)
                g = self.get_group(ir, sig)
                g.external_spec = st

            else:
                continue

        # 2) finalize groups: probabilities, sanity check
        # for sig, g in ir.groups.items():
        #     # choose to use probability summary or single probability:
        #     ps = prob_list.get(sig, [])
        #     if len(ps) == 1:
        #         g.probability = ps[0]
        #     else:
        #         g.probabilities_summary = _prob_summary(ps)

        #     # sanity check
        #     if g.fact_count == 0 and g.rule_count == 0 and not g.has_external and not g.has_agent:
        #         raise ValueError(f"for predicate {sig}, no facts, rules, or externals found.")

        try:
            return ir
        finally:
            pass
