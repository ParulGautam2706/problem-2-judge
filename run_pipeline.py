"""
End-to-end runner:

test suite
    -> pointwise judging
    -> validation
    -> position bias
    -> A/B comparison
    -> final report

Run:

    python run_pipeline.py
"""

import json
import os

from src.judge import JudgePipeline
from src.bias import run_position_bias_check
from src.aggregate import (
    pointwise_suite_report,
    compare_configs,
)
from src.validate import (
    agreement_with_gold,
    test_retest_consistency,
    adversarial_probe_report,
)


BASE = os.path.dirname(os.path.abspath(__file__))


def load_json(filename):
    path = os.path.join(BASE, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Required file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():

    # ---------------------------------------------------------
    # LOAD DATA
    # ---------------------------------------------------------

    suite = load_json("test_suite.json")
    ab = load_json("ab_comparison.json")

    # ---------------------------------------------------------
    # CREATE JUDGE
    # ---------------------------------------------------------

    pipeline = JudgePipeline()

    print("=" * 70)
    print("LLM-AS-JUDGE EVALUATION PIPELINE")
    print("=" * 70)

    print(
        f"\nJudge provider : {pipeline.judge_provider}"
    )

    print(
        f"Judge model    : {pipeline.llm.model}"
    )

    # ---------------------------------------------------------
    # 1. POINTWISE JUDGING
    # ---------------------------------------------------------

    print("\n== 1. Pointwise judging on test suite ==")

    verdicts = []

    for index, test_case in enumerate(suite, start=1):

        print(
            f"  Judging case {index}/{len(suite)}: "
            f"{test_case.get('id', index)}"
        )

        verdict = pipeline.judge_pointwise(test_case)

        verdicts.append(verdict)

    suite_report = pointwise_suite_report(verdicts)

    print(
        json.dumps(
            suite_report,
            indent=2,
            default=str,
        )
    )

    # ---------------------------------------------------------
    # 2. VALIDATION
    # ---------------------------------------------------------

    print("\n== 2. Judge validation ==")

    gold_labels = [
        tc.get("human_gold_score")
        for tc in suite
    ]

    agreement = agreement_with_gold(
        verdicts,
        gold_labels,
    )

    print("  Running test-retest consistency...")

    retest = test_retest_consistency(
        pipeline,
        suite[0],
        n_runs=3,
    )

    print("  Running adversarial probes...")

    adversarial = adversarial_probe_report(
        pipeline
    )

    validation_report = {
        "agreement_with_gold": agreement,
        "test_retest_consistency_sample": retest,
        "adversarial_probes": adversarial,
    }

    print(
        json.dumps(
            validation_report,
            indent=2,
            default=str,
        )
    )

    # ---------------------------------------------------------
    # 3. POSITION BIAS
    # ---------------------------------------------------------

    print("\n== 3. Position bias check ==")

    pairwise_cases = []

    for tc in ab["test_cases"]:

        case_id = tc["id"]

        if case_id not in ab["config_a_outputs"]:
            raise KeyError(
                f"Missing Config A output for {case_id}"
            )

        if case_id not in ab["config_b_outputs"]:
            raise KeyError(
                f"Missing Config B output for {case_id}"
            )

        pairwise_cases.append(
            {
                "id": case_id,
                "input": tc["input"],
                "output_a": ab["config_a_outputs"][case_id],
                "output_b": ab["config_b_outputs"][case_id],
            }
        )

    position_bias = run_position_bias_check(
        pipeline,
        pairwise_cases,
    )

    # Print summary only.
    position_summary = {
        "flip_rate": position_bias.get(
            "flip_rate",
            0.0,
        ),
        "stability_rate": position_bias.get(
            "stability_rate",
            0.0,
        ),
        "n_cases": position_bias.get(
            "n_cases",
            0,
        ),
        "n_stable": position_bias.get(
            "n_stable",
            0,
        ),
        "n_unstable": position_bias.get(
            "n_unstable",
            0,
        ),
    }

    print(
        json.dumps(
            position_summary,
            indent=2,
        )
    )

    # ---------------------------------------------------------
    # 4. A/B COMPARISON
    # ---------------------------------------------------------

    print(
        "\n== 4. A/B config comparison "
        "(terse prompt vs structured prompt) =="
    )

    ab_result = compare_configs(
        pipeline,
        ab["test_cases"],
        ab["config_a_outputs"],
        ab["config_b_outputs"],
    )

    ab_summary = {
        key: value
        for key, value in ab_result.items()
        if key != "per_case"
    }

    print(
        json.dumps(
            ab_summary,
            indent=2,
            default=str,
        )
    )

    # ---------------------------------------------------------
    # 5. SELF-ENHANCEMENT CHECK
    # ---------------------------------------------------------

    generator_provider = os.getenv(
        "GENERATOR_PROVIDER",
        "anthropic",
    )

    if (
        pipeline.judge_provider.lower()
        == generator_provider.lower()
    ):
        self_enhancement_note = (
            "Judge and generator use the SAME provider. "
            "Self-enhancement risk is not mitigated."
        )

    else:
        self_enhancement_note = (
            f"Judge provider "
            f"({pipeline.judge_provider}) differs from "
            f"generator provider "
            f"({generator_provider}). "
            "Self-enhancement risk is reduced."
        )

    # ---------------------------------------------------------
    # 6. FINAL REPORT
    # ---------------------------------------------------------

    full_report = {
        "suite_report": suite_report,

        "validation": validation_report,

        "position_bias": position_bias,

        "ab_comparison": ab_result,

        "self_enhancement_mitigation": (
            self_enhancement_note
        ),

        "judge_config": {
            "provider": pipeline.judge_provider,
            "model": pipeline.llm.model,
        },
    }

    out_path = os.path.join(
        BASE,
        "report.json",
    )

    with open(
        out_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            full_report,
            f,
            indent=2,
            default=str,
        )

    print(
        f"\nFull report written to {out_path}"
    )

    print(
        "Raw judge prompts/responses logged to "
        f"{os.path.join(BASE, 'logs', 'judge_log.jsonl')}"
    )


if __name__ == "__main__":
    main()