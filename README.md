# LLM-as-Judge Evaluation Pipeline

Judges model outputs against an explicit rubric, with position/verbosity/self-enhancement/
sycophancy bias mitigations built in and measured, not just discussed.

## Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
# Optional: judge a different model family than whatever generated the outputs,
# to mitigate self-enhancement bias:
export JUDGE_PROVIDER=openai
export OPENAI_API_KEY=sk-...
python run_pipeline.py
```
Everything (judge provider/model, generator provider/model) is env-configured — no hardcoded secrets.

## Judging modes implemented
- **Pointwise** (`judge_pointwise`): scores one output against the rubric, 1-5 per criterion +
  overall. Reference-based when `expected_output` is present, reference-free otherwise. Use for
  regression tracking a single config over time.
- **Pairwise** (`judge_pairwise`): A-vs-B, per-criterion + overall winner. Use for A/B'ing two
  prompts/models where you care about relative quality, not an absolute scale.

## Rubric
`src/rubric.py` — correctness, faithfulness, completeness, instruction_following, tone, safety.
Each is scored 1-5 with a required rationale grounded in specific input/output evidence (the
judge prompt explicitly forbids scoring high just because the text sounds fluent).

## Bias handling — what's implemented and measured

| Bias | Mitigation in code | Where measured |
|---|---|---|
| Position (A/B order) | `judge_pairwise(..., swap_order=...)` runs both orders; `compare_configs` only declares a per-case winner when both orders agree, else calls it a tie | `run_position_bias_check` → `flip_rate` |
| Verbosity/length | Explicit "don't favor length" instruction in the pairwise prompt; padded-but-empty answer probe | `run_verbosity_probe` in `src/bias.py` |
| Self-enhancement | `JUDGE_PROVIDER` is configured independently from the generator's provider — set it to a different model family | logged in `report.json.self_enhancement_mitigation` |
| Sycophancy/style | Rubric forces per-criterion grounding; confidently-wrong probe answer | `run_sycophancy_probe` |
| Score clustering | Few-shot anchor examples in the rubric prompt (`SCALE_DESCRIPTION`) | `score_distribution()` in `src/bias.py` |

Run `python run_pipeline.py` to execute the position-bias check and both probes and see the
actual flip rate / fooled-or-not results for this run, in `report.json`.

## Judge validation
`src/validate.py` implements all three:
- **Agreement with gold**: `agreement_with_gold()` — raw agreement rate + Cohen's kappa (binarized
  pass/fail) + Pearson correlation, against `human_gold_score` in `test_suite.json`.
- **Test-retest consistency**: `test_retest_consistency()` re-runs the same case N times and
  reports whether the verdict is stable.
- **Adversarial probes**: the verbosity + sycophancy probes double as the adversarial set
  (verbose-but-wrong vs terse-but-correct traps).

## A/B comparison
`ab_comparison.json` has 5 inputs run through two prompt configs (Config A: terse, no format
instruction; Config B: explicit format + example + trade-off instruction). `compare_configs()`
in `src/aggregate.py` runs both orders per case and declares an overall winner — see
`report.json.ab_comparison.declared_winner`.

## Logging
Every judge call (prompt + raw response + parsed verdict + token counts) is appended to
`logs/judge_log.jsonl` for auditability/replay.

## Discussion
**How biased was it before vs after mitigation?** The position-bias check quantifies this
directly — `flip_rate` in `report.json` is the fraction of A/B cases where the winner changed
purely because of presentation order. Run the pipeline and compare that number to what you'd
expect from an unmitigated judge (single-order judging has no way to even detect this, which is
itself the point — you can't fix what you don't measure). Similarly, `run_verbosity_probe`
directly reports whether the concise-but-correct answer lost to the padded one.

**Would I let it gate a release?** Only for the parts with strong validation evidence:
pass/fail regression checks on pointwise scores where `test_retest_consistency` shows low
variance are reasonable to gate on. I would NOT let a single unmitigated pairwise call gate a
release — position bias alone can flip a "winner" call; use the both-orders-agree rule in
`compare_configs` (which already returns "tie" on disagreement) so a release only blocks on a
verdict that held up under order-swapping, not a single roll of the dice.
"# problem-2-judge" 
"# problem2-judge" 
