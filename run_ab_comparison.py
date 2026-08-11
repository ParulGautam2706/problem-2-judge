"""
Compare two configs (prompt v1 vs v2, or model A vs B) and declare a winner.
Also reports the position-bias flip rate for these same pairs as a sanity
check alongside the declared winner.

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python run_ab_comparison.py [path/to/ab_comparison.json]
"""
import json
import os
import sys

from src.judge import JudgePipeline
from src.aggregate import compare_configs
from src.bias import run_position_bias_check

BASE = os.path.dirname(os.path.abspath(__file__))


def main():
    ab_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "ab_comparison.json")
    with open(ab_path) as f:
        ab = json.load(f)

    pipeline = JudgePipeline()

    ab_result = compare_configs(pipeline, ab["test_cases"], ab["config_a_outputs"], ab["config_b_outputs"])

    pairwise_cases = [
        {"id": tc["id"], "input": tc["input"],
         "output_a": ab["config_a_outputs"][tc["id"]], "output_b": ab["config_b_outputs"][tc["id"]]}
        for tc in ab["test_cases"]
    ]
    position_bias = run_position_bias_check(pipeline, pairwise_cases)

    out = {"ab_comparison": ab_result, "position_bias": position_bias,
           "judge_config": {"provider": pipeline.judge_provider, "model": pipeline.llm.model}}
    out_path = os.path.join(BASE, "ab_report.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    print(json.dumps({k: v for k, v in ab_result.items() if k != "per_case"}, indent=2))
    print("Position bias flip_rate:", position_bias["flip_rate"])
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
