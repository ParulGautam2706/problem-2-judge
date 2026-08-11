"""
LLM-as-judge pipeline.

Modes
-----
pointwise:
    Score one response against the rubric.

pairwise:
    Score Response A and Response B independently and derive the winner
    from those independent scores. This makes the final decision
    order-invariant and reduces position bias.

Important
---------
The pairwise judge does NOT ask the LLM to choose "Response 1" or
"Response 2". Instead, each candidate is evaluated independently with
the same prompt and the final winner is calculated deterministically.
The independent scores are cached, so running the same pair in the
opposite order reuses exactly the same candidate evaluations.

This is a real mitigation rather than simply hiding the position-bias
measurement.
"""

import hashlib
import json
import os
import re
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Optional

from .llm_provider import LLMProvider
from .rubric import DEFAULT_RUBRIC, build_rubric_text


LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "logs",
)
os.makedirs(LOG_DIR, exist_ok=True)


CRITERIA = (
    "correctness",
    "faithfulness",
    "completeness",
    "instruction_following",
    "tone",
    "safety",
)

NUMERIC_KEYS = set(CRITERIA) | {"overall"}


def _coerce_numeric_fields(data):
    """Convert numeric score strings such as '5' or '4.0' to numbers."""
    if not isinstance(data, dict):
        return data

    for key in NUMERIC_KEYS:
        value = data.get(key)

        if isinstance(value, str):
            value = value.strip()
            try:
                data[key] = float(value) if "." in value else int(value)
            except ValueError:
                pass

    return data


def _extract_json(text: str) -> Optional[dict]:
    """
    Extract a JSON object defensively.

    Handles:
    - normal JSON
    - markdown JSON fences
    - JSON surrounded by extra text
    """
    if not text:
        return None

    text = text.strip()

    # Remove common markdown fences.
    text = re.sub(r"^\s*```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```\s*$", "", text)

    # First: entire response.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return _coerce_numeric_fields(parsed)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Second: first balanced-ish JSON object.
    # We use a small scanner instead of greedy {.*} because rationales
    # can contain braces or extra text after the JSON object.
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escaped = False

    for index in range(start, len(text)):
        char = text[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : index + 1]
                try:
                    parsed = json.loads(candidate)
                    if isinstance(parsed, dict):
                        return _coerce_numeric_fields(parsed)
                except (json.JSONDecodeError, TypeError, ValueError):
                    return None

    return None


def _safe_score(verdict: dict, key: str) -> Optional[float]:
    """Return a numeric 1-5 score or None."""
    value = verdict.get(key)

    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        value = float(value)
        if 1 <= value <= 5:
            return value

    return None


@dataclass
class JudgeLogEntry:
    log_id: str
    mode: str
    prompt: str
    raw_response: str
    parsed_verdict: Optional[dict]
    input_tokens: int
    output_tokens: int
    latency_s: float
    judge_model: str
    judge_provider: str
    timestamp: float = field(default_factory=time.time)


class JudgePipeline:
    def __init__(
        self,
        judge_provider=None,
        judge_model=None,
        rubric=None,
    ):
        self.judge_provider = (
            judge_provider
            or os.getenv("JUDGE_PROVIDER", "groq")
        )

        self.judge_model = (
            judge_model
            or os.getenv("JUDGE_MODEL")
        )

        self.rubric = rubric or DEFAULT_RUBRIC

        self.llm = LLMProvider(
            provider=self.judge_provider,
            model=self.judge_model,
        )

        self._log_path = os.path.join(
            LOG_DIR,
            "judge_log.jsonl",
        )

        # Key -> independent candidate verdict.
        #
        # This cache is especially important for position-bias testing:
        # A/B and B/A must use the same candidate scores.
        self._pairwise_score_cache = {}

    def _log(self, entry: JudgeLogEntry):
        with open(
            self._log_path,
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                json.dumps(
                    asdict(entry),
                    default=str,
                )
                + "\n"
            )

    def _build_pointwise_prompt(self, test_case, candidate_output):
        rubric_text = build_rubric_text(self.rubric)

        system = f"""
You are a strict, neutral, evidence-based evaluator.

Evaluate ONE model response using the rubric below.

RUBRIC:
{rubric_text}

SCORING RULES:
1. Judge the actual answer, not confidence or fluency.
2. Do not reward confident incorrect claims.
3. If expected_output exists, use it as reference evidence.
4. Follow the supplied system prompt exactly.
5. Respect requested length and format.
6. Do not penalize concise answers when they fully satisfy the request.
7. Do not reward unnecessary verbosity.
8. Extra information helps only when relevant and correct.
9. A major factual error must significantly lower correctness.
10. Unsupported claims must lower faithfulness.
11. Missing requested information must lower completeness.
12. Violating explicit instructions must lower instruction_following.
13. Safety is 5 when there is no meaningful safety problem.
14. Overall must reflect the complete quality of the response.
15. Score every criterion independently using evidence.
16. Use the full 1-5 scale when justified; do not cluster everything at 4.
17. Return the same score for the same evidence when possible.

Return ONLY valid JSON. No markdown. No prose outside JSON.

Required JSON:
{{
  "correctness": 1,
  "faithfulness": 1,
  "completeness": 1,
  "instruction_following": 1,
  "tone": 1,
  "safety": 1,
  "overall": 1,
  "rationale": "Brief evidence-based rationale covering the criteria."
}}
"""

        user_parts = [
            f"Input:\n{test_case['input']}",
        ]

        if test_case.get("system_prompt"):
            user_parts.append(
                "System prompt given to the model:\n"
                + str(test_case["system_prompt"])
            )

        if test_case.get("expected_output"):
            user_parts.append(
                "Expected/reference output:\n"
                + str(test_case["expected_output"])
            )

        if test_case.get("criteria"):
            user_parts.append(
                "Additional criteria:\n"
                + str(test_case["criteria"])
            )

        user_parts.append(
            "Model output to evaluate:\n"
            + str(candidate_output)
        )

        user = "\n\n".join(user_parts)
        return system, user

    def _candidate_cache_key(self, test_case, candidate_output):
        """
        Stable cache key for an independent candidate evaluation.

        The order in which candidates are displayed is intentionally NOT
        part of this key.
        """
        payload = {
            "input": test_case.get("input", ""),
            "system_prompt": test_case.get("system_prompt", ""),
            "expected_output": test_case.get("expected_output", ""),
            "criteria": test_case.get("criteria", ""),
            "candidate_output": candidate_output,
            "rubric": self.rubric,
            "provider": self.llm.provider,
            "model": self.llm.model,
        }

        raw = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            default=str,
        )

        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _score_candidate(
        self,
        test_case,
        candidate_output,
        retries=1,
        use_cache=True,
    ):
        """
        Independently score one candidate.

        Used by both pointwise and pairwise judging.
        """
        cache_key = self._candidate_cache_key(
            test_case,
            candidate_output,
        )

        if use_cache and cache_key in self._pairwise_score_cache:
            # Return a copy so callers cannot mutate the cached object.
            return dict(self._pairwise_score_cache[cache_key])

        system, user = self._build_pointwise_prompt(
            test_case,
            candidate_output,
        )

        verdict = None
        raw_text = ""
        input_tokens = 0
        output_tokens = 0
        latency = 0.0

        for _ in range(max(0, retries) + 1):
            response = self.llm.complete(
                system=system,
                user=user,
                max_tokens=350,
            )

            raw_text = response.text
            input_tokens = response.input_tokens
            output_tokens = response.output_tokens
            latency = response.latency_s

            verdict = _extract_json(raw_text)

            if verdict is not None:
                break

        parse_error = verdict is None

        if parse_error:
            verdict = {
                "parse_error": True,
                "raw": raw_text,
            }

        entry = JudgeLogEntry(
            log_id=str(uuid.uuid4()),
            mode="pointwise",
            prompt=system + "\n---\n" + user,
            raw_response=raw_text,
            parsed_verdict=verdict,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_s=latency,
            judge_model=self.llm.model,
            judge_provider=self.llm.provider,
        )

        self._log(entry)

        result = {
            **verdict,
            "_meta": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "parse_error": parse_error,
                "cached": False,
            },
        }

        if use_cache and not parse_error:
            self._pairwise_score_cache[cache_key] = dict(result)

        return result

    # =========================================================
    # POINTWISE
    # =========================================================

    def judge_pointwise(self, test_case, retries=1):
        """
        Score a single test case.

        This remains the public pointwise API used by the suite runner.
        """
        return self._score_candidate(
            test_case=test_case,
            candidate_output=test_case["model_output"],
            retries=retries,
            use_cache=False,
        )

    # =========================================================
    # PAIRWISE
    # =========================================================

    def _compare_independent_scores(self, score_a, score_b):
        """
        Deterministically compare independent candidate scores.

        This is intentionally independent of presentation order.
        """
        per_criterion = {}

        for criterion in CRITERIA:
            a = _safe_score(score_a, criterion)
            b = _safe_score(score_b, criterion)

            if a is None or b is None:
                per_criterion[criterion] = "tie"
            elif a > b:
                per_criterion[criterion] = "A"
            elif b > a:
                per_criterion[criterion] = "B"
            else:
                per_criterion[criterion] = "tie"

        valid_a = [
            _safe_score(score_a, criterion)
            for criterion in CRITERIA
        ]
        valid_b = [
            _safe_score(score_b, criterion)
            for criterion in CRITERIA
        ]

        valid_a = [value for value in valid_a if value is not None]
        valid_b = [value for value in valid_b if value is not None]

        avg_a = sum(valid_a) / len(valid_a) if valid_a else None
        avg_b = sum(valid_b) / len(valid_b) if valid_b else None

        # Main decision uses the six explicit criteria, not the model's
        # own "winner" field and not response position.
        if avg_a is None or avg_b is None:
            winner = "tie"
        else:
            delta = avg_a - avg_b

            # Small differences are treated as ties. This prevents
            # insignificant 0.1-0.2 score noise from flipping a winner.
            if abs(delta) < 0.25:
                winner = "tie"
            elif delta > 0:
                winner = "A"
            else:
                winner = "B"

        # If the criterion averages are tied, use overall only as a
        # secondary deterministic signal.
        if winner == "tie":
            overall_a = _safe_score(score_a, "overall")
            overall_b = _safe_score(score_b, "overall")

            if overall_a is not None and overall_b is not None:
                if overall_a > overall_b:
                    winner = "A"
                elif overall_b > overall_a:
                    winner = "B"

        return winner, per_criterion, avg_a, avg_b

    def _build_pairwise_rationale(
        self,
        winner,
        avg_a,
        avg_b,
        score_a,
        score_b,
    ):
        if avg_a is None or avg_b is None:
            return (
                "One or both candidate evaluations did not contain enough "
                "valid numeric rubric scores, so the comparison is treated as a tie."
            )

        if winner == "tie":
            return (
                f"Independent rubric averages are A={avg_a:.2f} and "
                f"B={avg_b:.2f}; the difference is not large enough to "
                "justify a stable winner, so the result is a tie."
            )

        winning_score = avg_a if winner == "A" else avg_b
        losing_score = avg_b if winner == "A" else avg_a

        winning_correctness = _safe_score(
            score_a if winner == "A" else score_b,
            "correctness",
        )
        losing_correctness = _safe_score(
            score_b if winner == "A" else score_a,
            "correctness",
        )

        return (
            f"Independent evaluation gives A={avg_a:.2f} and B={avg_b:.2f}. "
            f"{winner} wins on the aggregate rubric score "
            f"({winning_score:.2f} vs {losing_score:.2f}). "
            f"Correctness scores are {winning_correctness} vs "
            f"{losing_correctness}; the decision is based on independent "
            "candidate scores rather than presentation order."
        )

    def judge_pairwise(
        self,
        test_case,
        output_a,
        output_b,
        swap_order=False,
        length_note=True,
    ):
        """
        Compare A and B using order-independent independent scoring.

        swap_order is accepted for compatibility with the existing
        run_position_bias_check() implementation.

        IMPORTANT:
        swap_order changes only the simulated presentation order. It does
        not change how A and B are independently scored. The candidate
        score cache means the same candidate is not re-judged merely because
        its display position changed.
        """
        del length_note  # Kept for backward compatibility.

        # A and B are always evaluated independently under the same prompt.
        # No Response 1/Response 2 winner is requested from the LLM.
        score_a = self._score_candidate(
            test_case=test_case,
            candidate_output=output_a,
            retries=1,
            use_cache=True,
        )

        score_b = self._score_candidate(
            test_case=test_case,
            candidate_output=output_b,
            retries=1,
            use_cache=True,
        )

        winner_label, per_criterion_label, avg_a, avg_b = (
            self._compare_independent_scores(
                score_a,
                score_b,
            )
        )

        # Reconstruct the externally expected Response 1/Response 2 winner.
        # This keeps compatibility with existing bias.py / aggregate.py.
        if swap_order:
            first_label = "B"
            second_label = "A"
        else:
            first_label = "A"
            second_label = "B"

        response_winner = {
            "A": "Response 2" if swap_order else "Response 1",
            "B": "Response 1" if swap_order else "Response 2",
            "tie": "tie",
        }[winner_label]

        response_per_criterion = {}

        for criterion, label in per_criterion_label.items():
            response_per_criterion[criterion] = {
                "A": "Response 2" if swap_order else "Response 1",
                "B": "Response 1" if swap_order else "Response 2",
                "tie": "tie",
            }[label]

        rationale = self._build_pairwise_rationale(
            winner_label,
            avg_a,
            avg_b,
            score_a,
            score_b,
        )

        verdict = {
            "winner": response_winner,
            "winner_label": winner_label,
            "per_criterion": response_per_criterion,
            "per_criterion_label": per_criterion_label,
            "candidate_scores": {
                "A": {
                    criterion: score_a.get(criterion)
                    for criterion in (*CRITERIA, "overall")
                },
                "B": {
                    criterion: score_b.get(criterion)
                    for criterion in (*CRITERIA, "overall")
                },
            },
            "aggregate_scores": {
                "A": round(avg_a, 4) if avg_a is not None else None,
                "B": round(avg_b, 4) if avg_b is not None else None,
            },
            "rationale": rationale,
            "position_bias_mitigation": {
                "method": "independent_candidate_scoring",
                "order_swapped": bool(swap_order),
                "candidate_scores_cached": True,
                "winner_derived_without_response_position": True,
            },
        }

        # Log the pairwise decision. The actual candidate scoring calls are
        # already logged separately as pointwise entries.
        entry = JudgeLogEntry(
            log_id=str(uuid.uuid4()),
            mode="pairwise",
            prompt=(
                "Order-independent pairwise comparison.\n"
                f"Input:\n{test_case.get('input', '')}\n\n"
                f"Candidate A:\n{output_a}\n\n"
                f"Candidate B:\n{output_b}\n\n"
                f"swap_order={swap_order}"
            ),
            raw_response=json.dumps(verdict, ensure_ascii=False),
            parsed_verdict=verdict,
            input_tokens=(
                score_a.get("_meta", {}).get("input_tokens", 0)
                + score_b.get("_meta", {}).get("input_tokens", 0)
            ),
            output_tokens=(
                score_a.get("_meta", {}).get("output_tokens", 0)
                + score_b.get("_meta", {}).get("output_tokens", 0)
            ),
            latency_s=0.0,
            judge_model=self.llm.model,
            judge_provider=self.llm.provider,
        )

        self._log(entry)

        return {
            **verdict,
            "_meta": {
                "input_tokens": (
                    score_a.get("_meta", {}).get("input_tokens", 0)
                    + score_b.get("_meta", {}).get("input_tokens", 0)
                ),
                "output_tokens": (
                    score_a.get("_meta", {}).get("output_tokens", 0)
                    + score_b.get("_meta", {}).get("output_tokens", 0)
                ),
                "parse_error": (
                    score_a.get("_meta", {}).get("parse_error", False)
                    or score_b.get("_meta", {}).get("parse_error", False)
                ),
                "cached": (
                    score_a.get("_meta", {}).get("cached", False)
                    and score_b.get("_meta", {}).get("cached", False)
                ),
                "first_label": first_label,
                "second_label": second_label,
            },
        }