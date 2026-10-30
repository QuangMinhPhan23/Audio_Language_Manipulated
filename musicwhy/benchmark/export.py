"""Benchmark export utilities for MusicWhy.

Supports JSON, HuggingFace Datasets, and Croissant metadata formats,
plus stratified train/val/test split creation.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

logger = logging.getLogger(__name__)

Question = Dict[str, Any]


class BenchmarkExporter:
    """Export MusicWhy benchmark questions to various formats."""

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    @staticmethod
    def to_json(
        questions: List[Question],
        output_path: Union[str, Path],
    ) -> Path:
        """Write questions to a standard JSON file.

        Parameters
        ----------
        questions : list of Question
        output_path : str or Path

        Returns
        -------
        Path
            The written file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "benchmark": "MusicWhy",
            "version": "1.0",
            "n_questions": len(questions),
            "task_counts": _count_by_task(questions),
            "questions": questions,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

        logger.info("Exported %d questions to JSON: %s", len(questions), output_path)
        return output_path

    # ------------------------------------------------------------------
    # HuggingFace Datasets export
    # ------------------------------------------------------------------

    @staticmethod
    def to_huggingface(
        questions: List[Question],
        dataset_name: str,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Any:
        """Convert questions to a HuggingFace ``Dataset`` object.

        If *output_dir* is given, the dataset is also saved to disk in
        Arrow format.  Requires the ``datasets`` package.

        Parameters
        ----------
        questions : list of Question
        dataset_name : str
            Name for the dataset (used as the config name).
        output_dir : str or Path, optional
            Directory to save the Arrow dataset.

        Returns
        -------
        datasets.Dataset
        """
        try:
            from datasets import Dataset, Features, Value, Sequence as HFSeq
        except ImportError as exc:
            raise ImportError(
                "The 'datasets' package is required for HuggingFace export. "
                "Install it with: pip install datasets"
            ) from exc

        # Flatten questions into column-oriented records
        records = _flatten_for_hf(questions)

        features = Features({
            "question_id": Value("string"),
            "task_id": Value("int32"),
            "question_text": Value("string"),
            "audio_reference": Value("string"),
            "option_A": Value("string"),
            "option_B": Value("string"),
            "option_C": Value("string"),
            "option_D": Value("string"),
            "correct_answer": Value("string"),
            "reference_explanation": Value("string"),
            "metadata_json": Value("string"),
        })

        ds = Dataset.from_list(records, features=features)
        ds = ds.rename_column("question_id", "id")

        if output_dir is not None:
            out = Path(output_dir) / dataset_name
            out.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(out))
            logger.info("Saved HuggingFace dataset to %s", out)

        logger.info(
            "Created HuggingFace Dataset '%s' with %d rows", dataset_name, len(ds),
        )
        return ds

    # ------------------------------------------------------------------
    # Croissant metadata
    # ------------------------------------------------------------------

    @staticmethod
    def to_croissant(
        questions: List[Question],
        metadata: Dict[str, Any],
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Generate Croissant (ML Commons) metadata for NeurIPS Datasets & Benchmarks.

        Parameters
        ----------
        questions : list of Question
            The full benchmark question set.
        metadata : dict
            Must include at least ``name``, ``description``, ``url``, and
            ``license``.  Optional: ``creators``, ``version``, ``citation``.
        output_path : str or Path, optional
            If provided, the Croissant JSON-LD is written to this file.

        Returns
        -------
        dict
            The Croissant JSON-LD metadata document.
        """
        task_counts = _count_by_task(questions)
        total = len(questions)

        # Build Croissant JSON-LD
        croissant: Dict[str, Any] = {
            "@context": {
                "@vocab": "https://schema.org/",
                "sc": "https://schema.org/",
                "ml": "http://mlcommons.org/croissant/",
                "rai": "http://mlcommons.org/croissant/RAI/",
            },
            "@type": "sc:Dataset",
            "name": metadata.get("name", "MusicWhy"),
            "description": metadata.get(
                "description",
                "A benchmark for evaluating musical reasoning in audio language models.",
            ),
            "url": metadata.get("url", ""),
            "license": metadata.get("license", "CC-BY-4.0"),
            "version": metadata.get("version", "1.0.0"),
            "datePublished": metadata.get("date_published", ""),
            "creator": [
                {"@type": "sc:Person", "name": name}
                for name in metadata.get("creators", [])
            ],
            "distribution": [
                {
                    "@type": "sc:FileObject",
                    "name": "questions.json",
                    "contentUrl": metadata.get("data_url", ""),
                    "encodingFormat": "application/json",
                    "sha256": "",  # to be filled at release time
                },
            ],
            "recordSet": [
                {
                    "@type": "ml:RecordSet",
                    "name": "benchmark_questions",
                    "description": (
                        f"MusicWhy benchmark with {total} questions across "
                        f"{len(task_counts)} tasks."
                    ),
                    "field": [
                        _croissant_field("question_id", "sc:Text", "Unique question identifier"),
                        _croissant_field("task_id", "sc:Integer", "Task number (1-5)"),
                        _croissant_field("question_text", "sc:Text", "Full question text"),
                        _croissant_field("audio_reference", "sc:Text", "Path or URL to audio"),
                        _croissant_field("options", "sc:Text", "JSON-encoded answer options"),
                        _croissant_field("correct_answer", "sc:Text", "Correct answer key"),
                        _croissant_field("reference_explanation", "sc:Text", "Reference explanation"),
                        _croissant_field("metadata", "sc:Text", "JSON-encoded metadata"),
                    ],
                    "numRecords": total,
                },
            ],
            "ml:taskType": [
                "multiple-choice-qa",
                "open-ended-generation",
                "audio-understanding",
            ],
        }

        # Add citation if provided
        if "citation" in metadata:
            croissant["citation"] = metadata["citation"]

        # RAI (Responsible AI) annotations
        croissant["rai:dataCollection"] = metadata.get(
            "data_collection",
            "Audio from CC-licensed sources; reasoning annotations from LLMs "
            "validated by human annotators.",
        )

        if output_path is not None:
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, "w", encoding="utf-8") as f:
                json.dump(croissant, f, indent=2, ensure_ascii=False)
            logger.info("Wrote Croissant metadata to %s", out)

        return croissant

    # ------------------------------------------------------------------
    # Splits
    # ------------------------------------------------------------------

    @staticmethod
    def create_splits(
        questions: List[Question],
        val_ratio: float = 0.1,
        test_ratio: float = 0.9,
        seed: int = 42,
    ) -> Dict[str, List[Question]]:
        """Create stratified validation/test splits.

        Stratification is by ``task_id`` so each split has proportional
        representation of every task.

        Parameters
        ----------
        questions : list of Question
        val_ratio : float
            Fraction of questions in the validation split.
        test_ratio : float
            Fraction of questions in the test split.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        dict
            ``{"val": [...], "test": [...]}``

        Raises
        ------
        ValueError
            If ratios do not sum to 1.0 (within tolerance).
        """
        import random as _random

        if abs((val_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError(
                f"val_ratio ({val_ratio}) + test_ratio ({test_ratio}) must equal 1.0"
            )

        rng = _random.Random(seed)

        # Group by task for stratified splitting
        by_task: Dict[int, List[Question]] = defaultdict(list)
        for q in questions:
            by_task[q.get("task_id", 0)].append(q)

        val_split: List[Question] = []
        test_split: List[Question] = []

        for task_id in sorted(by_task.keys()):
            # Split by item group (the song, or song pair), so the text and audio variants of one
            # item never land in different splits
            groups: Dict[str, List[Question]] = defaultdict(list)
            for q in by_task[task_id]:
                groups[_split_group(q)].append(q)
            keys = sorted(groups)
            rng.shuffle(keys)

            n_val = max(1, round(len(keys) * val_ratio))
            # Ensure test gets at least 1 if there are questions
            n_val = min(n_val, len(keys) - 1) if len(keys) > 1 else 0

            for k in keys[:n_val]:
                val_split.extend(groups[k])
            for k in keys[n_val:]:
                test_split.extend(groups[k])

        logger.info(
            "Created splits: val=%d, test=%d (from %d total)",
            len(val_split), len(test_split), len(questions),
        )
        return {"val": val_split, "test": test_split}


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _split_group(q: Question) -> str:
    """Key that ties together every question about the same song (or song pair)."""
    meta = q.get("metadata") or {}
    if meta.get("song_id"):
        return str(meta["song_id"])
    if meta.get("song_a_id"):
        return "|".join(sorted([str(meta["song_a_id"]), str(meta.get("song_b_id", ""))]))
    return str(meta.get("group") or q.get("question_id"))


def _count_by_task(questions: List[Question]) -> Dict[int, int]:
    counts: Dict[int, int] = defaultdict(int)
    for q in questions:
        counts[q.get("task_id", 0)] += 1
    return dict(sorted(counts.items()))


def _flatten_for_hf(questions: List[Question]) -> List[Dict[str, Any]]:
    """Flatten question dicts into a columnar format for HuggingFace."""
    records: List[Dict[str, Any]] = []
    for q in questions:
        options = q.get("options") or {}
        audio_ref = q.get("audio_reference", "")
        if isinstance(audio_ref, dict):
            audio_ref = json.dumps(audio_ref, ensure_ascii=False)

        records.append({
            "question_id": q.get("question_id", ""),
            "task_id": q.get("task_id", 0),
            "question_text": q.get("question_text", ""),
            "audio_reference": str(audio_ref),
            "option_A": str(options.get("A", "")),
            "option_B": str(options.get("B", "")),
            "option_C": str(options.get("C", "")),
            "option_D": str(options.get("D", "")),
            "correct_answer": str(q.get("correct_answer", "")),
            "reference_explanation": str(q.get("reference_explanation", "")),
            "metadata_json": json.dumps(q.get("metadata", {}), ensure_ascii=False),
        })
    return records


def _croissant_field(
    name: str, data_type: str, description: str,
) -> Dict[str, str]:
    return {
        "@type": "ml:Field",
        "name": name,
        "dataType": data_type,
        "description": description,
    }
