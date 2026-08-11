import statistics


def pointwise_suite_report(verdicts: list, pass_threshold: int = 3) -> dict:
    """verdicts: list of judge_pointwise() outputs. 'Pass' = overall >= pass_threshold."""
    valid = [v for v in verdicts if not v.get("_meta", {}).get("parse_error")]
    parse_failures = len(verdicts) - len(valid)

    overall_scores = [v.get("overall") for v in valid if isinstance(v.get("overall"), (int, float))]
    passes = [1 for s in overall_scores if s >= pass_threshold]

    per_criterion_means = {}
    if valid:
        keys = [k for k in valid[0].keys() if k not in ("overall", "rationale", "_meta")]
        for k in keys:
            vals = [v.get(k) for v in valid if isinstance(v.get(k), (int, float))]
            if vals:
                per_criterion_means[k] = round(statistics.mean(vals), 3)

    return {
        "n_cases": len(verdicts),
        "n_parse_failures": parse_failures,
        "pass_rate": round(len(passes) / len(overall_scores), 3) if overall_scores else None,
        "mean_overall": round(statistics.mean(overall_scores), 3) if overall_scores else None,
        "per_criterion_means": per_criterion_means,
        "total_judge_tokens": sum(v.get("_meta", {}).get("input_tokens", 0) + v.get("_meta", {}).get("output_tokens", 0) for v in verdicts),
    }


def compare_configs(pipeline, test_cases: list, config_a_outputs: dict, config_b_outputs: dict) -> dict:
    """test_cases: list of {id, input, system_prompt?}. config_*_outputs: {case_id: output_text}.
    Runs pairwise judging both orders per case (position-bias-controlled) and declares an overall winner."""
    wins_a = wins_b = ties = 0
    per_case = []
    for case in test_cases:
        cid = case["id"]
        out_a = config_a_outputs[cid]
        out_b = config_b_outputs[cid]
        v1 = pipeline.judge_pairwise(case, out_a, out_b, swap_order=False)
        v2 = pipeline.judge_pairwise(case, out_a, out_b, swap_order=True)
        w1, w2 = v1.get("winner_label"), v2.get("winner_label")

        if w1 == w2 and w1 in ("A", "B"):
            final = w1
        elif w1 == "tie" or w2 == "tie" or w1 != w2:
            final = "tie"  # disagreement across order -> don't force a call
        else:
            final = "tie"

        if final == "A":
            wins_a += 1
        elif final == "B":
            wins_b += 1
        else:
            ties += 1
        per_case.append({"case_id": cid, "order_ab_winner": w1, "order_ba_winner": w2, "final": final})

    n = len(test_cases)
    winner = "A" if wins_a > wins_b else ("B" if wins_b > wins_a else "tie")
    return {
        "n_cases": n,
        "config_a_wins": wins_a,
        "config_b_wins": wins_b,
        "ties": ties,
        "win_rate_a": round(wins_a / n, 3) if n else None,
        "win_rate_b": round(wins_b / n, 3) if n else None,
        "declared_winner": winner,
        "per_case": per_case,
    }
