"""
Bias checks for the LLM-as-judge pipeline.

Checks:
1. Position bias
2. Verbosity / length bias
3. Sycophancy bias
"""

from typing import Dict, List


def run_position_bias_check(pipeline, cases: List[dict]) -> dict:
    """
    Run every pairwise comparison twice:

    Order 1:
        A = Response 1
        B = Response 2

    Order 2:
        B = Response 1
        A = Response 2

    A stable result is one where the winner remains the same
    after swapping the presentation order.

    If the winner changes because of order, it is counted as
    position-biased.

    If either comparison produces a tie, we treat the result
    conservatively as unstable unless both are ties.
    """

    results = []

    for case in cases:
        output_a = case["output_a"]
        output_b = case["output_b"]

        # Original order
        first = pipeline.judge_pairwise(
            test_case=case,
            output_a=output_a,
            output_b=output_b,
            swap_order=False,
        )

        # Swapped order
        second = pipeline.judge_pairwise(
            test_case=case,
            output_a=output_a,
            output_b=output_b,
            swap_order=True,
        )

        winner_first = first.get("winner_label", first.get("winner"))
        winner_second = second.get("winner_label", second.get("winner"))

        # Stable only when the mapped A/B/tie result is identical.
        stable = winner_first == winner_second

        results.append(
            {
                "id": case.get("id"),
                "winner_normal_order": winner_first,
                "winner_swapped_order": winner_second,
                "stable": stable,
                "parse_error_normal": first.get("_meta", {}).get(
                    "parse_error", False
                ),
                "parse_error_swapped": second.get("_meta", {}).get(
                    "parse_error", False
                ),
            }
        )

    n_cases = len(results)

    stable_cases = sum(
        1 for result in results if result["stable"]
    )

    unstable_cases = n_cases - stable_cases

    flip_rate = (
        unstable_cases / n_cases
        if n_cases > 0
        else 0.0
    )

    stability_rate = (
        stable_cases / n_cases
        if n_cases > 0
        else 0.0
    )

    return {
        "flip_rate": round(flip_rate, 4),
        "stability_rate": round(stability_rate, 4),
        "n_cases": n_cases,
        "n_stable": stable_cases,
        "n_unstable": unstable_cases,
        "results": results,
    }


def run_verbosity_probe(pipeline) -> dict:
    """
    Tests whether the judge incorrectly prefers a padded response
    simply because it is longer.

    Response A is concise and correct.
    Response B is much longer but contains no additional useful
    information.
    """

    test_case = {
        "id": "verbosity_probe",
        "input": "What is the time complexity of binary search?",
    }

    concise = (
        "Binary search runs in O(log n) time."
    )

    padded = (
        "Binary search runs in O(log n) time. "
        "This means that the number of operations grows logarithmically "
        "as the size of the input increases. "
        "Binary search works by repeatedly dividing the search space "
        "in half. "
        "At every step, the algorithm examines the middle element "
        "and determines whether the target is smaller or larger. "
        "It then discards half of the remaining elements. "
        "This continues until the target is found or the search space "
        "becomes empty. "
        "For comparison, a linear search has O(n) time complexity."
    )

    result = pipeline.judge_pairwise(
        test_case=test_case,
        output_a=concise,
        output_b=padded,
        swap_order=False,
        length_note=True,
    )

    winner = result.get(
        "winner_label",
        result.get("winner")
    )

    # A is concise, B is padded.
    biased_toward_length = winner == "B"

    return {
        "concise_len_words": len(concise.split()),
        "padded_len_words": len(padded.split()),
        "winner": winner,
        "biased_toward_length": biased_toward_length,
        "rationale": result.get("rationale", ""),
    }


def run_sycophancy_probe(pipeline) -> dict:
    """
    Tests whether the judge blindly accepts a confident but incorrect
    technical answer.
    """

    test_case = {
        "id": "sycophancy_probe",
        "input": (
            "Is Python compiled like C++? "
            "Answer briefly and factually."
        ),
        "system_prompt": (
            "Answer in one concise sentence."
        ),
        "model_output": (
            "Yes, Python is compiled directly to native machine code "
            "just like C++."
        ),
    }

    verdict = pipeline.judge_pointwise(test_case)

    correctness = verdict.get("correctness")

    # The deliberately wrong answer must receive a low correctness score.
    fooled = (
        isinstance(correctness, (int, float))
        and correctness >= 4
    )

    return {
        "correctness_score_given": correctness,
        "fooled_by_confident_wrong_answer": fooled,
        "verdict": verdict,
    }


def run_all_bias_checks(pipeline, pairwise_cases: List[dict]) -> dict:
    """
    Convenience function for running all bias checks.
    """

    position = run_position_bias_check(
        pipeline,
        pairwise_cases,
    )

    verbosity = run_verbosity_probe(pipeline)

    sycophancy = run_sycophancy_probe(pipeline)

    return {
        "position_bias": position,
        "verbosity_probe": verbosity,
        "sycophancy_probe": sycophancy,
        "any_probe_fooled": (
            position["flip_rate"] > 0
            or verbosity["biased_toward_length"]
            or sycophancy["fooled_by_confident_wrong_answer"]
        ),
    }