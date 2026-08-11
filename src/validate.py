"""
Validation utilities for the LLM-as-judge pipeline.

Validation includes:

1. Agreement with human gold scores
2. Cohen's kappa
3. Pearson correlation
4. Test-retest consistency
5. Adversarial bias probes
"""

import statistics
from collections import Counter


def cohens_kappa(labels_a: list, labels_b: list):
    """
    Manual Cohen's kappa implementation.

    labels_a and labels_b must be parallel categorical lists.
    """

    if len(labels_a) != len(labels_b):
        raise ValueError(
            "Cohen's kappa requires lists of equal length."
        )

    n = len(labels_a)

    if n == 0:
        return None

    agree = sum(
        1
        for a, b in zip(labels_a, labels_b)
        if a == b
    )

    po = agree / n

    categories = set(labels_a) | set(labels_b)

    count_a = Counter(labels_a)
    count_b = Counter(labels_b)

    pe = sum(
        (count_a.get(category, 0) / n)
        * (count_b.get(category, 0) / n)
        for category in categories
    )

    if pe == 1:
        return 1.0

    return round(
        (po - pe) / (1 - pe),
        4,
    )


def agreement_with_gold(
    judge_verdicts: list,
    gold_labels: list,
    key: str = "overall",
    threshold: int = 3,
) -> dict:
    """
    Compare judge scores with human gold scores.

    Scores >= threshold are treated as PASS.
    Scores < threshold are treated as FAIL.
    """

    if len(judge_verdicts) != len(gold_labels):
        raise ValueError(
            "judge_verdicts and gold_labels must have equal length."
        )

    judge_pass = []

    for verdict in judge_verdicts:
        score = verdict.get(key)

        if isinstance(score, (int, float)):
            judge_pass.append(
                1 if score >= threshold else 0
            )
        else:
            judge_pass.append(0)

    gold_pass = []

    for gold in gold_labels:
        if isinstance(gold, (int, float)):
            gold_pass.append(
                1 if gold >= threshold else 0
            )
        else:
            gold_pass.append(0)

    if gold_pass:
        agreement_rate = (
            sum(
                1
                for j, g in zip(judge_pass, gold_pass)
                if j == g
            )
            / len(gold_pass)
        )
    else:
        agreement_rate = None

    kappa = cohens_kappa(
        judge_pass,
        gold_pass,
    )

    correlation = None

    if (
        len(gold_labels) > 1
        and all(
            isinstance(g, (int, float))
            for g in gold_labels
        )
    ):
        judge_scores = [
            v.get(key)
            for v in judge_verdicts
        ]

        if (
            all(
                isinstance(s, (int, float))
                for s in judge_scores
            )
            and len(set(judge_scores)) > 1
            and len(set(gold_labels)) > 1
        ):
            correlation = round(
                _pearson(
                    judge_scores,
                    gold_labels,
                ),
                4,
            )

    return {
        "agreement_rate": (
            round(agreement_rate, 4)
            if agreement_rate is not None
            else None
        ),
        "cohens_kappa": kappa,
        "pearson_correlation": correlation,
        "n": len(gold_labels),
    }


def _pearson(x, y):
    """
    Pearson correlation coefficient.
    """

    if len(x) != len(y):
        raise ValueError(
            "Pearson correlation requires equal length lists."
        )

    n = len(x)

    if n == 0:
        return 0.0

    mx = statistics.mean(x)
    my = statistics.mean(y)

    numerator = sum(
        (xi - mx) * (yi - my)
        for xi, yi in zip(x, y)
    )

    denominator_x = (
        sum(
            (xi - mx) ** 2
            for xi in x
        )
        ** 0.5
    )

    denominator_y = (
        sum(
            (yi - my) ** 2
            for yi in y
        )
        ** 0.5
    )

    if denominator_x == 0 or denominator_y == 0:
        return 0.0

    return numerator / (
        denominator_x * denominator_y
    )


def test_retest_consistency(
    pipeline,
    test_case: dict,
    n_runs: int = 3,
    key: str = "overall",
) -> dict:
    """
    Run the same evaluation multiple times.

    Stable scores indicate a more reliable judge.
    """

    verdicts = []

    for _ in range(n_runs):
        try:
            verdicts.append(
                pipeline.judge_pointwise(test_case)
            )
        except Exception as exc:
            verdicts.append(
                {
                    "parse_error": True,
                    "error": str(exc),
                }
            )

    scores = [
        v.get(key)
        for v in verdicts
        if isinstance(
            v.get(key),
            (int, float),
        )
    ]

    if not scores:
        return {
            "n_runs": n_runs,
            "scores": [],
            "consistent": None,
            "score_range": None,
            "stdev": None,
        }

    consistent = len(set(scores)) == 1

    return {
        "n_runs": n_runs,
        "scores": scores,
        "consistent": consistent,
        "score_range": max(scores) - min(scores),
        "stdev": (
            round(
                statistics.pstdev(scores),
                3,
            )
            if len(scores) > 1
            else 0.0
        ),
    }


def adversarial_probe_report(pipeline) -> dict:
    """
    Runs the verbosity and sycophancy adversarial probes.
    """

    from .bias import (
        run_verbosity_probe,
        run_sycophancy_probe,
    )

    verbosity = run_verbosity_probe(pipeline)

    sycophancy = run_sycophancy_probe(pipeline)

    return {
        "verbosity_probe": verbosity,
        "sycophancy_probe": sycophancy,
        "any_probe_fooled": (
            bool(
                verbosity.get(
                    "biased_toward_length",
                    False,
                )
            )
            or bool(
                sycophancy.get(
                    "fooled_by_confident_wrong_answer",
                    False,
                )
            )
        ),
    }