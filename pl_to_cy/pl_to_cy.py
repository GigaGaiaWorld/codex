#!/usr/bin/env python3
"""Lightweight Problog (RDF-style) to Neo4j Cypher converter.

Supported facts:
  unary_predicate(instance).
  binary_predicate(subject, object).

Unary predicates become entity labels, binary predicates become relationships.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class Fact:
    predicate: str
    args: Sequence[str]


def _strip_comments(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if "%" in line:
            line = line.split("%", 1)[0]
        lines.append(line)
    return "\n".join(lines)


def _split_args(arg_str: str) -> List[str]:
    args: List[str] = []
    current: List[str] = []
    in_quote = None
    escape = False
    for ch in arg_str:
        if escape:
            current.append(ch)
            escape = False
            continue
        if ch == "\\":
            escape = True
            current.append(ch)
            continue
        if ch in ("'", '"'):
            if in_quote is None:
                in_quote = ch
            elif in_quote == ch:
                in_quote = None
            current.append(ch)
            continue
        if ch == "," and in_quote is None:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(ch)
    if current:
        args.append("".join(current).strip())
    return [a for a in args if a]


def _split_facts(text: str) -> List[str]:
    facts: List[str] = []
    current: List[str] = []
    in_quote = None
    escape = False
    paren_depth = 0
    for ch in text:
        if escape:
            current.append(ch)
            escape = False
            continue
        if ch == "\\":
            escape = True
            current.append(ch)
            continue
        if ch in ("'", '"'):
            if in_quote is None:
                in_quote = ch
            elif in_quote == ch:
                in_quote = None
            current.append(ch)
            continue
        if in_quote is None:
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth = max(paren_depth - 1, 0)
            if ch == "." and paren_depth == 0:
                fact = "".join(current).strip()
                if fact:
                    facts.append(f"{fact}.")
                current = []
                continue
        current.append(ch)
    trailing = "".join(current).strip()
    if trailing:
        facts.append(trailing)
    return facts


def _parse_fact(raw: str) -> Fact | None:
    raw = raw.strip()
    if not raw:
        return None
    if not raw.endswith("."):
        raise ValueError(f"Fact missing terminating '.' -> {raw}")
    raw = raw[:-1].strip()
    if "(" not in raw or not raw.endswith(")"):
        raise ValueError(f"Invalid fact format -> {raw}")
    pred, rest = raw.split("(", 1)
    pred = pred.strip()
    args_str = rest[:-1]
    args = _split_args(args_str)
    if len(args) not in (1, 2):
        raise ValueError(f"Only unary or binary predicates supported -> {raw}")
    return Fact(predicate=pred, args=args)


def parse_facts(text: str) -> List[Fact]:
    cleaned = _strip_comments(text)
    parts = _split_facts(cleaned)
    facts: List[Fact] = []
    for part in parts:
        fact = _parse_fact(part)
        if fact:
            facts.append(fact)
    return facts


def _cypher_literal(value: str) -> str:
    value = value.strip()
    if (value.startswith("'") and value.endswith("'")) or (
        value.startswith('"') and value.endswith('"')
    ):
        value = value[1:-1]
    escaped = value.replace("\\", "\\\\").replace("'", "\\'")
    return f"'{escaped}'"


def _cypher_identifier(value: str) -> str:
    value = value.strip()
    escaped = value.replace("`", "``")
    return f"`{escaped}`"


def _emit_nodes(facts: Iterable[Fact]) -> List[str]:
    cypher = []
    seen = set()
    for fact in facts:
        if len(fact.args) != 1:
            continue
        instance = fact.args[0]
        label = _cypher_identifier(fact.predicate)
        key = (instance, label)
        if key in seen:
            continue
        seen.add(key)
        literal = _cypher_literal(instance)
        cypher.append(f"MERGE (n:Entity {{id: {literal}}})")
        cypher.append(f"SET n:{label}")
    return cypher


def _emit_relationships(facts: Iterable[Fact]) -> List[str]:
    cypher = []
    for fact in facts:
        if len(fact.args) != 2:
            continue
        subj, obj = fact.args
        rel = _cypher_identifier(fact.predicate)
        subj_lit = _cypher_literal(subj)
        obj_lit = _cypher_literal(obj)
        cypher.append(f"MERGE (s:Entity {{id: {subj_lit}}})")
        cypher.append(f"MERGE (o:Entity {{id: {obj_lit}}})")
        cypher.append(f"MERGE (s)-[:{rel}]->(o)")
    return cypher


def convert(text: str) -> str:
    facts = parse_facts(text)
    lines: List[str] = []
    lines.extend(_emit_nodes(facts))
    lines.extend(_emit_relationships(facts))
    return "\n".join(lines) + ("\n" if lines else "")


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Convert RDF-style Problog facts into Neo4j Cypher."
    )
    parser.add_argument("input", help="Path to .pl/.problog input file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output .cypher file (defaults to stdout)",
    )
    args = parser.parse_args(argv)

    text = ""
    with open(args.input, "r", encoding="utf-8") as handle:
        text = handle.read()

    cypher = convert(text)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(cypher)
    else:
        sys.stdout.write(cypher)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
