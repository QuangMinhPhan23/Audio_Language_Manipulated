"""Human validation utilities for MusicWhy benchmark questions.

Provides export/import for annotation workflows, inter-annotator agreement
(Krippendorff's alpha), and filtering based on agreement thresholds.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------
Question = Dict[str, Any]
Annotation = Dict[str, Any]
"""Expected annotation dict keys:
    question_id, annotator_id, selected_answer (for MC), rating (1-5 for open),
    optional free_text
"""


class HumanValidation:
    """Manages the human annotation lifecycle for benchmark questions.

    Typical workflow::

        hv = HumanValidation()
        hv.export_for_annotation(questions, "annotations_batch_1.json")
        # ... annotators work on the file ...
        annotations = hv.import_annotations("annotations_batch_1_done.json")
        alpha = hv.compute_agreement(annotations)
        good_questions = hv.filter_by_agreement(questions, annotations, threshold=0.6)
    """

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_for_annotation(
        self,
        questions: List[Question],
        output_path: Union[str, Path],
        format: str = "json",
    ) -> Path:
        """Export questions in a format suitable for human annotators.

        Parameters
        ----------
        questions : list of Question
            Benchmark questions to annotate.
        output_path : str or Path
            Destination file.
        format : ``"json"`` or ``"csv"``.

        Returns
        -------
        Path
            The written file path.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        annotation_items = self._prepare_annotation_items(questions)

        if format == "json":
            self._export_json(annotation_items, output_path)
        elif format == "csv":
            self._export_csv(annotation_items, output_path)
        else:
            raise ValueError(f"Unsupported export format: {format!r}. Use 'json' or 'csv'.")

        logger.info(
            "Exported %d questions for annotation to %s (%s)",
            len(annotation_items), output_path, format,
        )
        return output_path

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------

    def import_annotations(
        self,
        annotation_path: Union[str, Path],
    ) -> List[Annotation]:
        """Load completed annotations from a file.

        Auto-detects JSON or CSV based on file extension.

        Parameters
        ----------
        annotation_path : str or Path
            File produced by annotators.

        Returns
        -------
        list of Annotation
        """
        path = Path(annotation_path)
        if not path.exists():
            raise FileNotFoundError(f"Annotation file not found: {path}")

        if path.suffix.lower() == ".csv":
            annotations = self._import_csv(path)
        else:
            annotations = self._import_json(path)

        logger.info("Imported %d annotations from %s", len(annotations), path)
        return annotations

    # ------------------------------------------------------------------
    # Agreement
    # ------------------------------------------------------------------

    def compute_agreement(
        self,
        annotations: List[Annotation],
    ) -> float:
        """Compute Krippendorff's alpha for inter-annotator agreement.

        Supports nominal data (multiple-choice answers) and ordinal data
        (1-5 ratings) simultaneously.  When both ``selected_answer`` and
        ``rating`` are present, ``selected_answer`` is used for MC tasks
        and ``rating`` for open-ended tasks.

        Parameters
        ----------
        annotations : list of Annotation
            Each annotation must have ``question_id`` and ``annotator_id``,
            plus ``selected_answer`` and/or ``rating``.

        Returns
        -------
        float
            Krippendorff's alpha in [-1, 1].  Values above 0.667 are
            typically considered acceptable; above 0.8 is good.
        """
        # Build reliability matrix: rows = annotators, cols = questions (units)
        by_question: Dict[str, Dict[str, Any]] = {}
        for ann in annotations:
            qid = ann["question_id"]
            aid = ann["annotator_id"]
            value = ann.get("selected_answer") or ann.get("rating")
            if value is None:
                continue
            by_question.setdefault(qid, {})[aid] = value

        if not by_question:
            logger.warning("No valid annotations to compute agreement on.")
            return 0.0

        # Convert to coincidence matrix approach
        return self._krippendorff_alpha(by_question)

    # ------------------------------------------------------------------
    # Filter
    # ------------------------------------------------------------------

    def filter_by_agreement(
        self,
        questions: List[Question],
        annotations: List[Annotation],
        threshold: float = 0.6,
    ) -> List[Question]:
        """Keep only questions where annotators sufficiently agree.

        For each question, pairwise agreement among annotators is computed.
        Questions whose per-item agreement ratio falls below *threshold*
        are discarded.

        Parameters
        ----------
        questions : list of Question
        annotations : list of Annotation
        threshold : float
            Minimum fraction of annotator pairs that must agree.

        Returns
        -------
        list of Question
            Filtered questions.
        """
        by_question: Dict[str, List[Any]] = {}
        for ann in annotations:
            qid = ann["question_id"]
            value = ann.get("selected_answer") or ann.get("rating")
            if value is not None:
                by_question.setdefault(qid, []).append(value)

        passing_ids: set[str] = set()
        for qid, values in by_question.items():
            if len(values) < 2:
                # Only one annotation -- cannot assess agreement, keep by default
                passing_ids.add(qid)
                continue
            agreement_ratio = self._pairwise_agreement(values)
            if agreement_ratio >= threshold:
                passing_ids.add(qid)

        filtered = [q for q in questions if q.get("question_id") in passing_ids]
        logger.info(
            "Filtered %d -> %d questions at agreement threshold %.2f",
            len(questions), len(filtered), threshold,
        )
        return filtered

    # ------------------------------------------------------------------
    # Private helpers -- export/import
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_annotation_items(questions: List[Question]) -> List[Dict[str, Any]]:
        """Strip reference answers and add annotation fields."""
        items: List[Dict[str, Any]] = []
        for q in questions:
            item: Dict[str, Any] = {
                "question_id": q.get("question_id", ""),
                "task_id": q.get("task_id"),
                "question_text": q.get("question_text", ""),
                "audio_reference": q.get("audio_reference", ""),
                "options": q.get("options"),
                # Annotation fields (to be filled by annotator)
                "annotator_id": "",
                "selected_answer": "",
                "rating": None,
                "free_text": "",
            }
            items.append(item)
        return items

    @staticmethod
    def _export_json(items: List[Dict[str, Any]], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {"version": "1.0", "items": items},
                f, indent=2, ensure_ascii=False,
            )

    @staticmethod
    def _export_csv(items: List[Dict[str, Any]], path: Path) -> None:
        if not items:
            return
        fieldnames = [
            "question_id", "task_id", "question_text", "audio_reference",
            "options", "annotator_id", "selected_answer", "rating", "free_text",
        ]
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for item in items:
                row = dict(item)
                # Serialize options dict to JSON string for CSV
                if isinstance(row.get("options"), dict):
                    row["options"] = json.dumps(row["options"], ensure_ascii=False)
                writer.writerow(row)

    @staticmethod
    def _import_json(path: Path) -> List[Annotation]:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            return payload.get("items", payload.get("annotations", []))
        return []

    @staticmethod
    def _import_csv(path: Path) -> List[Annotation]:
        annotations: List[Annotation] = []
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ann: Annotation = dict(row)
                # Parse rating back to int if present
                if ann.get("rating") not in (None, "", "None"):
                    try:
                        ann["rating"] = int(ann["rating"])
                    except (ValueError, TypeError):
                        try:
                            ann["rating"] = float(ann["rating"])
                        except (ValueError, TypeError):
                            pass
                annotations.append(ann)
        return annotations

    # ------------------------------------------------------------------
    # Private helpers -- agreement metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _pairwise_agreement(values: List[Any]) -> float:
        """Fraction of annotator pairs that gave the same value."""
        n = len(values)
        if n < 2:
            return 1.0
        agree = 0
        total = 0
        for i in range(n):
            for j in range(i + 1, n):
                total += 1
                if values[i] == values[j]:
                    agree += 1
        return agree / total if total > 0 else 1.0

    @staticmethod
    def _krippendorff_alpha(
        by_question: Dict[str, Dict[str, Any]],
    ) -> float:
        """Compute Krippendorff's alpha using the coincidence-matrix method.

        Handles nominal data.  For ordinal data (ratings), values are treated
        as nominal categories; for a full ordinal distance function, use a
        dedicated library (e.g., ``krippendorff``).

        Parameters
        ----------
        by_question : dict
            ``{question_id: {annotator_id: value, ...}, ...}``

        Returns
        -------
        float
            Alpha coefficient.
        """
        # Collect all value pairs within each unit (question)
        value_counts: Dict[Any, float] = {}
        observed_disagreement = 0.0
        n_pairable = 0.0  # total number of pairable values

        for qid, annotator_values in by_question.items():
            values = list(annotator_values.values())
            m = len(values)
            if m < 2:
                continue

            # Count pairs within this unit
            for i in range(m):
                for j in range(i + 1, m):
                    if values[i] != values[j]:
                        observed_disagreement += 1.0
                    # We weight each pair by 1/(m-1) per Krippendorff
                n_pairable += m

            # Update marginal value counts
            for v in values:
                value_counts[v] = value_counts.get(v, 0.0) + 1.0

        if n_pairable < 2:
            return 1.0  # Perfect agreement by default (too few data)

        # Total number of observed pairable judgments
        total_pairs = sum(
            len(vs) * (len(vs) - 1) / 2
            for vs in (
                list(av.values()) for av in by_question.values()
            )
            if len(vs) >= 2
        )

        if total_pairs == 0:
            return 1.0

        # Normalize observed disagreement
        d_o = observed_disagreement / total_pairs

        # Expected disagreement from marginal distribution
        n_total = sum(value_counts.values())
        expected_disagreement = 0.0
        all_values = list(value_counts.keys())
        for i in range(len(all_values)):
            for j in range(i + 1, len(all_values)):
                n_c = value_counts[all_values[i]]
                n_k = value_counts[all_values[j]]
                expected_disagreement += n_c * n_k

        d_e = expected_disagreement / (n_total * (n_total - 1) / 2) if n_total > 1 else 0.0

        if d_e == 0.0:
            return 1.0  # All annotators used the same value

        alpha = 1.0 - (d_o / d_e)
        return alpha
