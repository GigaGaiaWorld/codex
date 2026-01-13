# Auto-Analyst (DSPy)

This project recreates the Auto-Analyst agent system described in the Firebird Technologies article, using DSPy and a lightweight tool registry. The agent plans analyses, selects tools, and synthesizes answers for structured datasets.

## Features

- **Planning agent**: generates a short analysis plan based on the question and dataset overview.
- **Tool selection agent**: chooses analysis tools (summary stats, top categories, correlations, plots).
- **Answer synthesis**: produces a final response grounded in tool outputs.
- **Fallback mode**: runs deterministic analysis when an LLM is not configured.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set your model credentials (OpenAI example):

```bash
export OPENAI_API_KEY=your_key
export OPENAI_MODEL=gpt-4o-mini
```

## Run

```bash
python -m auto_analyst.cli path/to/data.csv "What drives revenue by region?"
```

Artifacts are written to `outputs/analysis_report.md` and plot images (if requested).
