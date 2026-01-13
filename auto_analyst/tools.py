from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pandas as pd


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[..., Any]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def list_descriptions(self) -> str:
        return "\n".join(
            f"- {tool.name}: {tool.description}" for tool in self._tools.values()
        )

    def run(self, name: str, payload: dict[str, Any]) -> Any:
        if name not in self._tools:
            raise ValueError(f"Unknown tool: {name}")
        return self._tools[name].handler(**payload)


class DataTools:
    def __init__(self, df: pd.DataFrame, output_dir: Path) -> None:
        self.df = df
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def dataset_overview(self) -> dict[str, Any]:
        return {
            "rows": int(self.df.shape[0]),
            "columns": list(self.df.columns),
            "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            "missing_values": self.df.isna().sum().to_dict(),
        }

    def summary_stats(self) -> dict[str, Any]:
        numeric = self.df.select_dtypes(include="number")
        if numeric.empty:
            return {"summary": "No numeric columns found."}
        return numeric.describe().to_dict()

    def top_categories(self, column: str, n: int = 5) -> dict[str, Any]:
        counts = self.df[column].value_counts().head(n)
        return {"column": column, "top_counts": counts.to_dict()}

    def correlation_matrix(self) -> dict[str, Any]:
        numeric = self.df.select_dtypes(include="number")
        if numeric.empty:
            return {"correlation": "No numeric columns found."}
        return numeric.corr().to_dict()

    def save_plot(self, x: str, y: str, kind: str = "scatter") -> dict[str, Any]:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        if kind == "scatter":
            ax.scatter(self.df[x], self.df[y])
        elif kind == "line":
            ax.plot(self.df[x], self.df[y])
        else:
            raise ValueError("Unsupported plot kind")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        plot_path = self.output_dir / f"{kind}_{x}_vs_{y}.png"
        fig.savefig(plot_path, bbox_inches="tight")
        plt.close(fig)
        return {"plot": str(plot_path)}


def parse_tool_input(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("tool_input must be valid JSON") from exc
