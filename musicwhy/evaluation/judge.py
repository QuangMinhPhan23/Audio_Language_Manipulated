"""LLM-as-judge evaluation for MusicWhy open-ended responses.

Uses GPT-4o as the judge model with temperature=0, with optional majority
voting over multiple samples to improve reliability.
"""

from __future__ import annotations

import logging
import os
import statistics
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default rubrics and criteria
# ---------------------------------------------------------------------------

_DEFAULT_OPEN_RUBRIC = """\
You are an expert music analyst and psychoacoustician evaluating a model's
answer to a music reasoning question.

## Scoring rubric (1-5):
1 = Completely wrong or irrelevant; no connection to the audio or question.
2 = Mostly wrong with minor relevant points; significant misconceptions.
3 = Partially correct; identifies some relevant aspects but misses key reasoning
    or makes notable errors.
4 = Mostly correct; good reasoning with minor gaps or imprecisions.
5 = Excellent; accurate, specific, well-reasoned, references concrete audio
    features and causal mechanisms.

## Evaluation task
Reference answer (ground truth):
{reference}

Model's answer:
{prediction}

Provide your evaluation in the following JSON format:
{{"score": <1-5>, "rationale": "<brief explanation of the score>"}}
"""

_EXPLANATION_CRITERIA = {
    "factual_accuracy": (
        "Are the factual claims about audio features, historical events, "
        "and cultural context correct?"
    ),
    "causal_coherence": (
        "Does the explanation establish a clear and logical causal chain "
        "from audio features to the claimed effect or connection?"
    ),
    "specificity": (
        "Does the explanation reference specific, concrete audio details "
        "(tempo values, spectral characteristics, instrument choices) "
        "rather than making vague or generic claims?"
    ),
}

_EXPLANATION_RUBRIC = """\
You are an expert evaluating the quality of a music-context explanation.

## Reference explanation:
{reference}

## Model's explanation:
{prediction}

## Evaluate on these criteria (score each 1-5):
{criteria_text}

## Scoring rubric for each criterion (1-5):
1 = No evidence of this quality.
2 = Minimal evidence; mostly lacking.
3 = Moderate; some relevant elements present.
4 = Good; clear evidence with minor gaps.
5 = Excellent; strong, detailed evidence throughout.

Respond in JSON format:
{{
    "scores": {{{criteria_keys_template}}},
    "rationale": "<brief overall rationale>"
}}
"""


# ---------------------------------------------------------------------------
# LLMJudge
# ---------------------------------------------------------------------------

class LLMJudge:
    """GPT-4o-based judge for scoring open-ended model responses.

    Parameters
    ----------
    model_name : str
        OpenAI model to use as judge.
    temperature : float
        Sampling temperature (0 for deterministic).
    n_samples : int
        Number of independent judge calls per item; final score is the
        majority (mode) for integer scores or the mean.
    api_key : str, optional
        OpenAI API key.  Falls back to ``OPENAI_API_KEY`` env var.
    """

    def __init__(
        self,
        model_name: str = "gpt-4o",
        temperature: float = 0.0,
        n_samples: int = 1,
        api_key: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.temperature = temperature
        self.n_samples = max(1, n_samples)
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client: Any = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        import openai

        if not self._api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is required for the LLM judge."
            )
        self._client = openai.OpenAI(api_key=self._api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_open_ended(
        self,
        prediction: str,
        reference: str,
        rubric: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Score a single open-ended response against a reference.

        Parameters
        ----------
        prediction : str
            Model's response.
        reference : str
            Ground-truth or reference answer.
        rubric : str, optional
            Custom rubric template.  Must contain ``{reference}`` and
            ``{prediction}`` placeholders.

        Returns
        -------
        dict
            ``{"score": int (1-5), "rationale": str}``
        """
        rubric = rubric or _DEFAULT_OPEN_RUBRIC
        prompt = rubric.format(reference=reference, prediction=prediction)
        results = self._call_judge_n(prompt)

        scores = [r.get("score", 3) for r in results]
        rationales = [r.get("rationale", "") for r in results]

        final_score = self._majority_score(scores)
        return {
            "score": final_score,
            "rationale": rationales[0] if len(rationales) == 1 else (
                f"Majority score {final_score} from {len(scores)} samples. "
                f"Sample rationale: {rationales[0]}"
            ),
            "all_scores": scores,
        }

    def evaluate_explanation(
        self,
        prediction: str,
        reference: str,
        criteria: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Score an explanation on multiple criteria.

        Parameters
        ----------
        prediction : str
        reference : str
        criteria : list of str, optional
            Criterion names from ``_EXPLANATION_CRITERIA``.
            Defaults to all three: factual_accuracy, causal_coherence,
            specificity.

        Returns
        -------
        dict
            ``{"scores": {criterion: int}, "overall": float, "rationale": str}``
        """
        if criteria is None:
            criteria = list(_EXPLANATION_CRITERIA.keys())

        criteria_text = "\n".join(
            f"- **{c}**: {_EXPLANATION_CRITERIA.get(c, c)}" for c in criteria
        )
        criteria_keys_template = ", ".join(f'"{c}": <1-5>' for c in criteria)

        prompt = _EXPLANATION_RUBRIC.format(
            reference=reference,
            prediction=prediction,
            criteria_text=criteria_text,
            criteria_keys_template=criteria_keys_template,
        )

        results = self._call_judge_n(prompt)

        # Aggregate per-criterion scores across samples
        aggregated: Dict[str, List[int]] = {c: [] for c in criteria}
        rationales: List[str] = []
        for r in results:
            raw_scores = r.get("scores", {})
            for c in criteria:
                aggregated[c].append(int(raw_scores.get(c, 3)))
            rationales.append(r.get("rationale", ""))

        final_scores = {
            c: self._majority_score(vals) for c, vals in aggregated.items()
        }
        overall = sum(final_scores.values()) / len(final_scores) if final_scores else 0.0

        return {
            "scores": final_scores,
            "overall": round(overall, 2),
            "rationale": rationales[0] if rationales else "",
        }

    def batch_judge(
        self,
        predictions: List[str],
        references: List[str],
        rubric: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Score a batch of prediction-reference pairs.

        Parameters
        ----------
        predictions : list of str
        references : list of str
        rubric : str, optional

        Returns
        -------
        list of dict
            One result dict per pair.
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Mismatched lengths: {len(predictions)} predictions vs "
                f"{len(references)} references."
            )
        results: List[Dict[str, Any]] = []
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            try:
                result = self.evaluate_open_ended(pred, ref, rubric)
            except Exception as exc:
                logger.error("Judge failed on item %d: %s", i, exc)
                result = {"score": 0, "rationale": f"Evaluation error: {exc}"}
            results.append(result)
            if (i + 1) % 50 == 0:
                logger.info("Judged %d / %d items", i + 1, len(predictions))

        logger.info("Batch judging complete: %d items", len(results))
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _call_judge_n(self, prompt: str) -> List[Dict[str, Any]]:
        """Call the judge model *n_samples* times and collect parsed results."""
        results: List[Dict[str, Any]] = []
        for _ in range(self.n_samples):
            raw = self._call_judge(prompt)
            parsed = self._parse_judge_response(raw)
            results.append(parsed)
        return results

    def _call_judge(self, prompt: str) -> str:
        """Single LLM judge call."""
        self._ensure_client()
        response = self._client.chat.completions.create(
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=500,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a precise evaluation judge. Always respond "
                        "with valid JSON only, no extra text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content or ""

    @staticmethod
    def _parse_judge_response(raw: str) -> Dict[str, Any]:
        """Parse the JSON response from the judge model."""
        import json

        # Strip markdown code fences if present
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning("Failed to parse judge response as JSON: %s", text[:200])
            # Attempt to extract score from text
            score = 3  # default
            for token in text.split():
                try:
                    val = int(token)
                    if 1 <= val <= 5:
                        score = val
                        break
                except ValueError:
                    continue
            return {"score": score, "rationale": f"[parse fallback] {text[:200]}"}

    @staticmethod
    def _majority_score(scores: List[int]) -> int:
        """Return the most common score (mode), breaking ties by median."""
        if not scores:
            return 3
        if len(scores) == 1:
            return scores[0]
        try:
            return statistics.mode(scores)
        except statistics.StatisticsError:
            # Multi-modal: use median as tiebreaker
            return round(statistics.median(scores))
