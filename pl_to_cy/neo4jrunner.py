#!/usr/bin/env python3
"""Run Cypher from a file against Neo4j using the official driver."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, Sequence

try:
    from neo4j import GraphDatabase
except ImportError as exc:  # pragma: no cover - runtime guard
    raise SystemExit(
        "Missing dependency: install the 'neo4j' Python package to use neo4jrunner."
    ) from exc


def _read_statements(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read()
    statements = [stmt.strip() for stmt in content.split(";") if stmt.strip()]
    for stmt in statements:
        yield stmt


def run_cypher(uri: str, user: str, password: str, cypher_path: str) -> None:
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        with driver.session() as session:
            for statement in _read_statements(cypher_path):
                session.run(statement)
    finally:
        driver.close()


def main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description="Run a Cypher file via Neo4j driver.")
    parser.add_argument("cypher", help="Path to .cypher file")
    parser.add_argument(
        "--uri",
        default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j bolt URI (default from NEO4J_URI)",
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("NEO4J_USER", "neo4j"),
        help="Neo4j username (default from NEO4J_USER)",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("NEO4J_PASSWORD", "neo4j"),
        help="Neo4j password (default from NEO4J_PASSWORD)",
    )
    args = parser.parse_args(argv)

    run_cypher(args.uri, args.user, args.password, args.cypher)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
