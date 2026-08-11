"""
Run the pointwise suite judging + validation.

Usage:
    python run_suite_report.py
    python run_suite_report.py path/to/test_suite.json
"""

import json
import os
import sys

from src.judge import JudgePipeline
from src.aggregate import pointwise_suite_report
from src.validate import (
    agreement_with_gold,
    test_retest_consistency,
    adversarial_probe_report,
)


BASE = os.path.dirname(
    os.path.abspath(__file__)
)


def main():
    suite_path = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(
            BASE,
            "test_suite.json",
        )
    )

    if not os.path.isfile(suite_path):
        raise FileNotFoundError(
            f"Test suite not found: {suite_path}"
        )

    with open(
        suite_path,
        "r",
        encoding="utf-8",
    ) as f:
        suite = json.load(f)

    if not isinstance(suite, list):
        raise ValueError(
            "test_suite.json must contain a JSON array"
        )

    if not suite:
        raise ValueError(
            "test_suite.json contains no test cases"
        )

    pipeline = JudgePipeline()

    verdicts = [
        pipeline.judge_pointwise(test_case)
        for test_case in suite
    ]

    suite_report = pointwise_suite_report(
        verdicts
    )

    gold_labels = [
        test_case.get("human_gold_score")
        for test_case in suite
    ]

    validation = {
        "agreement_with_gold": agreement_with_gold(
            verdicts,
            gold_labels,
        ),
        "test_retest_consistency_sample":
            test_retest_consistency(
                pipeline,
                suite[0],
                n_runs=3,
            ),
        "adversarial_probes":
            adversarial_probe_report(
                pipeline
            ),
    }

    out = {
        "suite_report": suite_report,
        "validation": validation,
        "verdicts": verdicts,
        "judge_config": {
            "provider": pipeline.judge_provider,
            "model": pipeline.llm.model,
        },
    }

    out_path = os.path.join(
        BASE,
        "suite_report.json",
    )

    with open(
        out_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            out,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print(
        json.dumps(
            {
                "suite_report": suite_report,
                "validation": validation,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    )

    print(
        f"\nWritten to {out_path}"
    )


if __name__ == "__main__":
    main()