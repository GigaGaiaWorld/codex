from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import dspy
import pandas as pd

from auto_analyst.signatures import PlanAnalysis, SelectTool, SynthesizeAnswer
from auto_analyst.tools import DataTools, Tool, ToolRegistry, parse_tool_input


class AutoAnalyst:
    def __init__(self, output_dir: str = "outputs") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._configure_dspy()
        self.planner = dspy.ChainOfThought(PlanAnalysis)
        self.tool_selector = dspy.Predict(SelectTool)
        self.synthesizer = dspy.ChainOfThought(SynthesizeAnswer)

    def _configure_dspy(self) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        if api_key:
            lm = dspy.LM(model=model, api_key=api_key)
            dspy.configure(lm=lm)

    def run(self, dataset_path: str, question: str) -> dict[str, Any]:
        df = self._load_dataset(dataset_path)
        tools = DataTools(df, self.output_dir)
        registry = ToolRegistry()
        registry.register(
            Tool(
                name="dataset_overview",
                description="Summarize rows, columns, dtypes, missing values.",
                handler=tools.dataset_overview,
            )
        )
        registry.register(
            Tool(
                name="summary_stats",
                description="Describe numeric columns with count, mean, std, min, max.",
                handler=tools.summary_stats,
            )
        )
        registry.register(
            Tool(
                name="top_categories",
                description="Top category counts for a column.",
                handler=tools.top_categories,
            )
        )
        registry.register(
            Tool(
                name="correlation_matrix",
                description="Correlation matrix for numeric columns.",
                handler=tools.correlation_matrix,
            )
        )
        registry.register(
            Tool(
                name="save_plot",
                description="Save a scatter or line plot for two columns.",
                handler=tools.save_plot,
            )
        )

        overview = tools.dataset_overview()
        plan = self._make_plan(question, overview)
        results: list[dict[str, Any]] = []
        if plan:
            results = self._run_tool_loop(registry, plan)

        if not results:
            results = self._fallback_results(registry)

        answer = self._synthesize_answer(question, results)
        report_path = self._write_report(question, plan, results, answer)
        return {
            "plan": plan,
            "results": results,
            "answer": answer,
            "report_path": str(report_path),
        }

    def _load_dataset(self, dataset_path: str) -> pd.DataFrame:
        path = Path(dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")
        if path.suffix == ".csv":
            return pd.read_csv(path)
        if path.suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        raise ValueError("Unsupported file type. Use CSV or Parquet.")

    def _make_plan(self, question: str, overview: dict[str, Any]) -> str:
        if not dspy.settings.lm:
            return ""
        response = self.planner(
            question=question,
            dataset_overview=json.dumps(overview, indent=2),
        )
        return response.analysis_plan.strip()

    def _run_tool_loop(self, registry: ToolRegistry, plan: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        available = registry.list_descriptions()
        for _ in range(3):
            response = self.tool_selector(
                analysis_plan=plan,
                available_tools=available,
                prior_results=json.dumps(results, indent=2),
            )
            tool_name = response.tool_name.strip()
            tool_input = parse_tool_input(response.tool_input)
            output = registry.run(tool_name, tool_input)
            results.append({"tool": tool_name, "input": tool_input, "output": output})
        return results

    def _fallback_results(self, registry: ToolRegistry) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for tool_name in ["dataset_overview", "summary_stats", "correlation_matrix"]:
            output = registry.run(tool_name, {})
            results.append({"tool": tool_name, "input": {}, "output": output})
        return results

    def _synthesize_answer(self, question: str, results: list[dict[str, Any]]) -> str:
        if not dspy.settings.lm:
            return (
                "LLM not configured. Review the report for dataset overview, summary "
                "statistics, and correlations."
            )
        response = self.synthesizer(
            question=question,
            analysis_results=json.dumps(results, indent=2),
        )
        return response.answer.strip()

    def _write_report(
        self,
        question: str,
        plan: str,
        results: list[dict[str, Any]],
        answer: str,
    ) -> Path:
        report = self.output_dir / "analysis_report.md"
        report.write_text(
            "\n".join(
                [
                    "# Auto-Analyst Report",
                    f"**Question:** {question}",
                    "",
                    "## Plan",
                    plan or "(No plan generated)",
                    "",
                    "## Tool Results",
                    json.dumps(results, indent=2),
                    "",
                    "## Answer",
                    answer,
                ]
            ),
            encoding="utf-8",
        )
        return report
