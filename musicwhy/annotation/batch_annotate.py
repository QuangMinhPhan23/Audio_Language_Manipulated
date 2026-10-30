"""
musicwhy.annotation.batch_annotate
===================================

BatchAnnotator -- orchestrates the full Phase 3 reasoning annotation
pipeline:

1. Load the track manifest (JSON list of track metadata).
2. Load pre-extracted features (one JSON per track, or aggregate Parquet).
3. Load the world-context database (year/country -> context text).
4. For every track:
   a. Translate features to natural language.
   b. Look up world context for (year, country).
   c. Call the LLM reasoning generator.
   d. Optionally cross-validate with a second model.
5. Save per-track annotation JSON files + aggregate Parquet.
6. Track progress, costs, and quality metrics.
7. Resume from checkpoint if interrupted.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for :class:`BatchAnnotator`.

    Parameters
    ----------
    manifest_path : str
        Path to the JSON manifest (list of track dicts).
    features_dir : str
        Directory containing per-track feature JSON files, named
        ``{track_id}.json``.
    world_context_path : str
        Path to a JSON file mapping ``"{year}_{country}"`` keys to
        context description strings.
    output_dir : str
        Where per-track annotation JSONs are saved.
    checkpoint_path : str
        Path to the checkpoint file tracking completed track ids.
    aggregate_parquet_path : str
        Path for the final aggregate Parquet file.
    cross_validate : bool
        Whether to run Approach 3 (cross-validation) with a second model.
    second_provider : str
        Provider for the cross-validation model (``"openai"``, etc.).
    second_model_name : str
        Model id for the cross-validation model.
    concurrency : int
        Number of concurrent async annotation tasks.
    """

    manifest_path: str = "manifest.json"
    features_dir: str = "features"
    world_context_path: str = "world_context.json"
    output_dir: str = "annotations"
    checkpoint_path: str = "annotations/checkpoint.json"
    aggregate_parquet_path: str = "annotations/annotations.parquet"
    cross_validate: bool = False
    second_provider: str = "openai"
    second_model_name: str = "gpt-4o"
    concurrency: int = 5


class BatchAnnotator:
    """Orchestrate the full reasoning annotation pipeline.

    Parameters
    ----------
    config : BatchConfig
        Paths and pipeline settings.
    reasoning_generator : ReasoningGenerator
        Primary generator (Approach 1).
    second_generator : ReasoningGenerator, optional
        Used only when ``config.cross_validate`` is ``True``.
    """

    def __init__(
        self,
        config: BatchConfig,
        reasoning_generator: Any,  # ReasoningGenerator
        second_generator: Optional[Any] = None,
    ) -> None:
        self.config = config
        self._primary = reasoning_generator
        self._secondary = second_generator

        # Runtime state
        self._completed: set[str] = set()
        self._failed: Dict[str, str] = {}
        self._quality_scores: List[float] = []
        self._start_time: float = 0.0

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    @staticmethod
    def load_manifest(path: str) -> List[Dict[str, Any]]:
        """Load the track manifest JSON.

        Expected format::

            [
                {
                    "track_id": "fma_012345",
                    "title": "Song Name",
                    "artist": "Artist",
                    "year": 2018,
                    "country": "US",
                    "genre": "electronic",
                    ...
                },
                ...
            ]
        """
        manifest_path = Path(path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {path}")
        with open(manifest_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError("Manifest must be a JSON array of track objects")
        logger.info("Loaded manifest: %d tracks", len(data))
        return data

    @staticmethod
    def load_features(features_dir: str, track_id: str) -> Dict[str, Any]:
        """Load per-track feature JSON.

        Looks for ``{features_dir}/{track_id}.json``.
        """
        fpath = Path(features_dir) / f"{track_id}.json"
        if not fpath.exists():
            raise FileNotFoundError(f"Features not found: {fpath}")
        with open(fpath, "r", encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def load_all_features(features_dir: str) -> Dict[str, Dict[str, Any]]:
        """Load every feature JSON in the directory into a dict."""
        result: Dict[str, Dict[str, Any]] = {}
        fdir = Path(features_dir)
        if not fdir.is_dir():
            logger.warning("Features directory does not exist: %s", features_dir)
            return result
        for fpath in fdir.glob("*.json"):
            track_id = fpath.stem
            try:
                with open(fpath, "r", encoding="utf-8") as fh:
                    result[track_id] = json.load(fh)
            except Exception as exc:
                logger.warning("Could not load %s: %s", fpath, exc)
        logger.info("Loaded features for %d tracks", len(result))
        return result

    @staticmethod
    def load_world_context(path: str) -> Dict[str, str]:
        """Load the world-context lookup table.

        Expected format::

            {
                "2018_US": "In 2018, the US economy grew 2.9% ...",
                "2020_global": "The COVID-19 pandemic ...",
                ...
            }
        """
        ctx_path = Path(path)
        if not ctx_path.exists():
            logger.warning(
                "World context file not found: %s. "
                "Annotations will lack historical context.",
                path,
            )
            return {}
        with open(ctx_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        logger.info("Loaded world context: %d entries", len(data))
        return data

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> None:
        """Resume from a previous checkpoint if one exists."""
        cp_path = Path(self.config.checkpoint_path)
        if cp_path.exists():
            with open(cp_path, "r", encoding="utf-8") as fh:
                cp = json.load(fh)
            self._completed = set(cp.get("completed", []))
            self._failed = cp.get("failed", {})
            logger.info(
                "Resumed from checkpoint: %d completed, %d failed",
                len(self._completed),
                len(self._failed),
            )
        else:
            self._completed = set()
            self._failed = {}

    def _save_checkpoint(self) -> None:
        """Persist current progress to disk."""
        cp_path = Path(self.config.checkpoint_path)
        cp_path.parent.mkdir(parents=True, exist_ok=True)
        cp = {
            "completed": sorted(self._completed),
            "failed": self._failed,
            "cost_summary": self._primary.cost_summary(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(cp_path, "w", encoding="utf-8") as fh:
            json.dump(cp, fh, indent=2)

    # ------------------------------------------------------------------
    # Metadata formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_metadata(track: Dict[str, Any]) -> str:
        """Build a human-readable metadata string from a manifest entry."""
        parts: List[str] = []
        if track.get("title"):
            parts.append(f"Title: {track['title']}")
        if track.get("artist"):
            parts.append(f"Artist: {track['artist']}")
        if track.get("year"):
            parts.append(f"Year: {track['year']}")
        if track.get("country"):
            parts.append(f"Country: {track['country']}")
        if track.get("genre"):
            parts.append(f"Genre: {track['genre']}")
        if track.get("album"):
            parts.append(f"Album: {track['album']}")
        return "\n".join(parts) if parts else "(no metadata)"

    @staticmethod
    def _world_context_key(track: Dict[str, Any]) -> str:
        """Build the lookup key for the world-context database."""
        year = track.get("year", "")
        country = track.get("country", "global")
        return f"{year}_{country}"

    # ------------------------------------------------------------------
    # Single-track annotation
    # ------------------------------------------------------------------

    def _annotate_one(
        self,
        track: Dict[str, Any],
        features: Dict[str, Any],
        world_context_db: Dict[str, str],
        stem_features: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Run the full annotation pipeline on a single track.

        Returns a dict ready to be serialised as the per-track JSON.
        """
        from musicwhy.annotation.llm_reasoning import (
            features_to_description,
            stem_comparison_description,
        )

        track_id = track.get("track_id", "unknown")
        metadata_str = self._format_metadata(track)
        feature_desc = features_to_description(features)

        # Build stem descriptions if stem features are available
        stem_desc_parts: List[str] = []
        if stem_features:
            for stem_name, s_features in stem_features.items():
                stem_desc_parts.append(
                    stem_comparison_description(features, s_features, stem_name)
                )
        stem_desc_str = "\n".join(stem_desc_parts) if stem_desc_parts else "(no stem data available)"

        # Look up world context
        ctx_key = self._world_context_key(track)
        world_ctx = world_context_db.get(ctx_key, "")
        if not world_ctx:
            # Try year-only fallback
            year_key = f"{track.get('year', '')}_global"
            world_ctx = world_context_db.get(year_key, "(no world context available)")

        # --- Primary reasoning (Approach 1) ---
        primary_reasoning = self._primary.generate_from_features(
            song_metadata=metadata_str,
            feature_description=feature_desc,
            stem_descriptions=stem_desc_str,
            world_context=world_ctx,
        )

        result: Dict[str, Any] = {
            "track_id": track_id,
            "metadata": track,
            "feature_description": feature_desc,
            "stem_descriptions": stem_desc_str,
            "world_context": world_ctx,
            "primary_reasoning": primary_reasoning,
            "primary_model": self._primary.config.model_name,
            "cross_validation": None,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        # --- Cross-validation (Approach 3) ---
        if self.config.cross_validate and self._secondary is not None:
            try:
                secondary_reasoning = self._secondary.generate_from_features(
                    song_metadata=metadata_str,
                    feature_description=feature_desc,
                    stem_descriptions=stem_desc_str,
                    world_context=world_ctx,
                )
                cv_result = self._primary.cross_validate(
                    primary_reasoning, secondary_reasoning
                )
                result["cross_validation"] = {
                    "secondary_model": self._secondary.config.model_name,
                    "secondary_reasoning": secondary_reasoning,
                    **cv_result,
                }
                # Track quality via cross-validation confidence
                confidence = cv_result.get("confidence", 0.0)
                if isinstance(confidence, (int, float)):
                    self._quality_scores.append(float(confidence))
            except Exception as exc:
                logger.warning(
                    "Cross-validation failed for %s: %s", track_id, exc
                )
                result["cross_validation"] = {"error": str(exc)}

        return result

    async def _annotate_one_async(
        self,
        track: Dict[str, Any],
        features: Dict[str, Any],
        world_context_db: Dict[str, str],
        stem_features: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Async wrapper for single-track annotation."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            self._annotate_one,
            track,
            features,
            world_context_db,
            stem_features,
        )

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------

    def _save_annotation(
        self, annotation: Dict[str, Any], output_dir: str
    ) -> Path:
        """Write a per-track annotation JSON."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        track_id = annotation["track_id"]
        fpath = out_path / f"{track_id}.json"
        with open(fpath, "w", encoding="utf-8") as fh:
            json.dump(annotation, fh, indent=2, ensure_ascii=False)
        return fpath

    def _save_aggregate_parquet(
        self, annotations: List[Dict[str, Any]]
    ) -> Path:
        """Save all annotations as a single Parquet file.

        Flattens the nested dicts into a tabular form suitable for
        analytics.
        """
        import pandas as pd

        rows: List[Dict[str, Any]] = []
        for ann in annotations:
            row: Dict[str, Any] = {
                "track_id": ann["track_id"],
                "title": ann.get("metadata", {}).get("title", ""),
                "artist": ann.get("metadata", {}).get("artist", ""),
                "year": ann.get("metadata", {}).get("year", None),
                "country": ann.get("metadata", {}).get("country", ""),
                "genre": ann.get("metadata", {}).get("genre", ""),
                "feature_description": ann.get("feature_description", ""),
                "world_context": ann.get("world_context", ""),
                "primary_reasoning": ann.get("primary_reasoning", ""),
                "primary_model": ann.get("primary_model", ""),
                "timestamp": ann.get("timestamp", ""),
            }
            cv = ann.get("cross_validation")
            if cv and isinstance(cv, dict) and "error" not in cv:
                row["cv_confidence"] = cv.get("confidence", None)
                row["cv_summary"] = cv.get("summary", "")
                row["cv_agreed_claims_count"] = len(
                    cv.get("agreed_claims", [])
                )
                row["cv_disputed_claims_count"] = len(
                    cv.get("disputed_claims", [])
                )
                row["secondary_model"] = cv.get("secondary_model", "")
            else:
                row["cv_confidence"] = None
                row["cv_summary"] = ""
                row["cv_agreed_claims_count"] = 0
                row["cv_disputed_claims_count"] = 0
                row["secondary_model"] = ""
            rows.append(row)

        df = pd.DataFrame(rows)
        out_path = Path(self.config.aggregate_parquet_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.info(
            "Saved aggregate Parquet (%d rows) to %s", len(df), out_path
        )
        return out_path

    # ------------------------------------------------------------------
    # Progress / quality reporting
    # ------------------------------------------------------------------

    def progress_report(self, total: int) -> Dict[str, Any]:
        """Return a snapshot of annotation progress and quality."""
        elapsed = time.monotonic() - self._start_time if self._start_time else 0
        done = len(self._completed)
        remaining = total - done
        rate = done / elapsed if elapsed > 0 else 0
        eta_seconds = remaining / rate if rate > 0 else float("inf")

        return {
            "completed": done,
            "failed": len(self._failed),
            "total": total,
            "progress_pct": round(100 * done / total, 1) if total else 0,
            "elapsed_seconds": round(elapsed, 1),
            "tracks_per_minute": round(rate * 60, 2),
            "eta_seconds": round(eta_seconds, 1) if eta_seconds != float("inf") else None,
            "avg_cv_confidence": (
                round(float(np.mean(self._quality_scores)), 3)
                if self._quality_scores
                else None
            ),
            "cost": self._primary.cost_summary(),
        }

    # ------------------------------------------------------------------
    # Main entry points
    # ------------------------------------------------------------------

    def run(
        self,
        stem_features_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run the full batch annotation pipeline synchronously.

        Parameters
        ----------
        stem_features_dir : str, optional
            Directory containing per-stem feature JSONs named
            ``{track_id}_{stem_name}.json``.

        Returns
        -------
        list of dict
            All annotations produced (including previously checkpointed).
        """
        self._start_time = time.monotonic()

        manifest = self.load_manifest(self.config.manifest_path)
        world_ctx = self.load_world_context(self.config.world_context_path)
        self._load_checkpoint()

        all_annotations: List[Dict[str, Any]] = []
        total = len(manifest)

        for idx, track in enumerate(manifest):
            track_id = track.get("track_id", f"track_{idx}")

            if track_id in self._completed:
                logger.debug("Skipping %s (already done)", track_id)
                # Load the existing annotation for aggregation
                existing_path = (
                    Path(self.config.output_dir) / f"{track_id}.json"
                )
                if existing_path.exists():
                    with open(existing_path, "r", encoding="utf-8") as fh:
                        all_annotations.append(json.load(fh))
                continue

            logger.info(
                "Annotating %s (%d/%d)...", track_id, idx + 1, total
            )

            try:
                features = self.load_features(
                    self.config.features_dir, track_id
                )
            except FileNotFoundError:
                logger.warning(
                    "No features for %s, skipping", track_id
                )
                self._failed[track_id] = "features_not_found"
                continue

            # Load stem features if available
            stem_feats: Optional[Dict[str, Dict[str, Any]]] = None
            if stem_features_dir:
                stem_feats = {}
                stem_dir = Path(stem_features_dir)
                for stem_path in stem_dir.glob(f"{track_id}_*.json"):
                    stem_name = stem_path.stem.replace(f"{track_id}_", "")
                    try:
                        with open(stem_path, "r", encoding="utf-8") as fh:
                            stem_feats[stem_name] = json.load(fh)
                    except Exception as exc:
                        logger.warning(
                            "Could not load stem %s: %s", stem_path, exc
                        )

            try:
                annotation = self._annotate_one(
                    track, features, world_ctx, stem_feats
                )
                self._save_annotation(annotation, self.config.output_dir)
                all_annotations.append(annotation)
                self._completed.add(track_id)
            except Exception as exc:
                logger.error("Failed to annotate %s: %s", track_id, exc)
                self._failed[track_id] = str(exc)

            # Checkpoint every 10 tracks
            if (idx + 1) % 10 == 0:
                self._save_checkpoint()
                report = self.progress_report(total)
                logger.info(
                    "Progress: %d/%d (%.1f%%) | Cost: $%.2f | "
                    "Rate: %.1f tracks/min",
                    report["completed"],
                    report["total"],
                    report["progress_pct"],
                    report["cost"]["total_cost_usd"],
                    report["tracks_per_minute"],
                )

        # Final checkpoint and aggregate
        self._save_checkpoint()

        if all_annotations:
            self._save_aggregate_parquet(all_annotations)

        report = self.progress_report(total)
        logger.info(
            "Batch annotation complete: %d/%d tracks annotated, "
            "%d failed. Total cost: $%.2f",
            report["completed"],
            report["total"],
            report["failed"],
            report["cost"]["total_cost_usd"],
        )

        return all_annotations

    async def run_async(
        self,
        stem_features_dir: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Run the batch pipeline with async concurrency.

        Uses ``asyncio.Semaphore`` to limit concurrent API calls to
        ``config.concurrency``.

        Parameters
        ----------
        stem_features_dir : str, optional
            Directory containing per-stem feature JSONs.

        Returns
        -------
        list of dict
            All annotations.
        """
        self._start_time = time.monotonic()

        manifest = self.load_manifest(self.config.manifest_path)
        world_ctx = self.load_world_context(self.config.world_context_path)
        self._load_checkpoint()

        semaphore = asyncio.Semaphore(self.config.concurrency)
        all_annotations: List[Dict[str, Any]] = []
        total = len(manifest)

        async def _process(
            idx: int, track: Dict[str, Any]
        ) -> Optional[Dict[str, Any]]:
            track_id = track.get("track_id", f"track_{idx}")

            if track_id in self._completed:
                existing_path = (
                    Path(self.config.output_dir) / f"{track_id}.json"
                )
                if existing_path.exists():
                    with open(existing_path, "r", encoding="utf-8") as fh:
                        return json.load(fh)
                return None

            try:
                features = self.load_features(
                    self.config.features_dir, track_id
                )
            except FileNotFoundError:
                self._failed[track_id] = "features_not_found"
                return None

            stem_feats: Optional[Dict[str, Dict[str, Any]]] = None
            if stem_features_dir:
                stem_feats = {}
                stem_dir = Path(stem_features_dir)
                for stem_path in stem_dir.glob(f"{track_id}_*.json"):
                    stem_name = stem_path.stem.replace(f"{track_id}_", "")
                    try:
                        with open(stem_path, "r", encoding="utf-8") as fh:
                            stem_feats[stem_name] = json.load(fh)
                    except Exception:
                        pass

            async with semaphore:
                try:
                    annotation = await self._annotate_one_async(
                        track, features, world_ctx, stem_feats
                    )
                    self._save_annotation(annotation, self.config.output_dir)
                    self._completed.add(track_id)

                    # Periodic checkpoint (roughly every 10)
                    if len(self._completed) % 10 == 0:
                        self._save_checkpoint()
                        report = self.progress_report(total)
                        logger.info(
                            "Progress: %d/%d (%.1f%%)",
                            report["completed"],
                            report["total"],
                            report["progress_pct"],
                        )

                    return annotation
                except Exception as exc:
                    logger.error(
                        "Failed to annotate %s: %s",
                        track_id,
                        exc,
                    )
                    self._failed[track_id] = str(exc)
                    return None

        tasks = [_process(idx, track) for idx, track in enumerate(manifest)]
        results = await asyncio.gather(*tasks)

        all_annotations = [r for r in results if r is not None]

        self._save_checkpoint()
        if all_annotations:
            self._save_aggregate_parquet(all_annotations)

        report = self.progress_report(total)
        logger.info(
            "Async batch annotation complete: %d/%d tracks, "
            "%d failed. Cost: $%.2f",
            report["completed"],
            report["total"],
            report["failed"],
            report["cost"]["total_cost_usd"],
        )

        return all_annotations
