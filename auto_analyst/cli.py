import argparse
import json

from auto_analyst.agent import AutoAnalyst


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Auto-Analyst on a dataset.")
    parser.add_argument("dataset", help="Path to a CSV or Parquet file.")
    parser.add_argument("question", help="Question to analyze.")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory to store reports and artifacts.",
    )
    args = parser.parse_args()

    analyst = AutoAnalyst(output_dir=args.output_dir)
    result = analyst.run(args.dataset, args.question)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
