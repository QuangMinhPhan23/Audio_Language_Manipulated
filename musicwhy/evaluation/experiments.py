"""Experiment runners for MusicWhy evaluation (Phase 5).

Five experiments, each as a standalone function:
  Exp 1 -- Audio LLM Benchmark Evaluation
  Exp 2 -- Raw Features vs. Model Embeddings
  Exp 3 -- Source-Aware Emotion Ablation
  Exp 4 -- Sound-World Correlation Analysis
  Exp 5 -- Instrumental vs. Vocal Performance Gap
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from musicwhy.evaluation.judge import LLMJudge
from musicwhy.evaluation.metrics import (
    bootstrap_ci,
    mc_accuracy,
    mean_judge_score,
    pearson_with_lag,
    performance_gap,
)
from musicwhy.evaluation.models import BaseModelWrapper, MERTWrapper

logger = logging.getLogger(__name__)

Question = Dict[str, Any]
SongRecord = Dict[str, Any]


# ---------------------------------------------------------------------------
# Experiment 1: Audio LLM Benchmark Evaluation
# ---------------------------------------------------------------------------

def run_exp1_benchmark(
    models: List[BaseModelWrapper],
    benchmark: List[Question],
    judge: LLMJudge,
    *,
    tasks: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Run all benchmark models on the MusicWhy question set.

    For multiple-choice tasks, computes accuracy.  For open-ended tasks,
    uses the LLM judge to score responses.

    Parameters
    ----------
    models : list of BaseModelWrapper
        Model wrappers to evaluate.
    benchmark : list of Question
        Full benchmark question set.
    judge : LLMJudge
        Judge instance for open-ended evaluation.
    tasks : list of int, optional
        Subset of task IDs to run (1-5).  Defaults to all.

    Returns
    -------
    dict
        Nested results: ``{model_name: {task_id: {metric: value}}}``
    """
    if tasks is None:
        tasks = [1, 2, 3, 4, 5]

    # Group questions by task
    by_task: Dict[int, List[Question]] = {}
    for q in benchmark:
        tid = q.get("task_id", 0)
        if tid in tasks:
            by_task.setdefault(tid, []).append(q)

    results: Dict[str, Dict[int, Dict[str, Any]]] = {}

    for model in models:
        model_name = model.name
        logger.info("Evaluating model: %s", model_name)
        results[model_name] = {}

        for tid in sorted(by_task.keys()):
            task_qs = by_task[tid]
            logger.info("  Task %d: %d questions", tid, len(task_qs))

            try:
                task_result = _evaluate_task(model, task_qs, tid, judge)
            except NotImplementedError:
                logger.warning(
                    "  Model %s does not support task %d; skipping.",
                    model_name, tid,
                )
                task_result = {"skipped": True, "reason": "not_implemented"}
            except Exception as exc:
                logger.error(
                    "  Error evaluating %s on task %d: %s", model_name, tid, exc,
                )
                task_result = {"error": str(exc)}

            results[model_name][tid] = task_result

    return results


def _evaluate_task(
    model: BaseModelWrapper,
    questions: List[Question],
    task_id: int,
    judge: LLMJudge,
) -> Dict[str, Any]:
    """Evaluate a single model on a single task's questions."""
    mc_preds: List[str] = []
    mc_truths: List[str] = []
    open_judge_results: List[Dict[str, Any]] = []

    for q in questions:
        audio = q.get("audio_reference", "")
        q_text = q.get("question_text", "")
        options = q.get("options")
        correct = q.get("correct_answer")
        ref_expl = q.get("reference_explanation", "")

        # Multiple-choice tasks (1, 2, 4)
        if options and correct:
            if task_id == 2:
                pred = model.predict_source(audio, q.get("metadata", {}).get("emotion", ""))
                # Normalize: the options are stem names
                mc_preds.append(pred.strip().lower())
                mc_truths.append(str(correct).strip().lower())
            else:
                pred = model.answer_mc(audio, q_text, options)
                mc_preds.append(pred)
                mc_truths.append(str(correct))

        # Open-ended tasks (3, 5) -- also generate open-ended for task 1
        if task_id in (3, 5) or (task_id == 1 and ref_expl):
            if task_id == 5:
                context = q.get("metadata", {}).get("world_context", "")
                response = model.explain_connection(audio, context, q_text)
            else:
                response = model.answer_open(audio, q_text)

            if ref_expl:
                if task_id in (3, 5):
                    score_result = judge.evaluate_explanation(response, ref_expl)
                else:
                    score_result = judge.evaluate_open_ended(response, ref_expl)
                score_result["prediction"] = response
                open_judge_results.append(score_result)

    result: Dict[str, Any] = {"n_questions": len(questions)}

    if mc_preds:
        acc = mc_accuracy(mc_preds, mc_truths)
        ci = bootstrap_ci([1.0 if p == g else 0.0 for p, g in zip(mc_preds, mc_truths)])
        result["mc_accuracy"] = acc
        result["mc_accuracy_ci"] = (ci["ci_lower"], ci["ci_upper"])
        result["mc_n"] = len(mc_preds)

    if open_judge_results:
        scores = [
            r.get("score") or r.get("overall", 0) for r in open_judge_results
        ]
        result["open_mean_score"] = float(np.mean(scores)) if scores else 0.0
        result["open_score_ci"] = (
            bootstrap_ci(scores)["ci_lower"],
            bootstrap_ci(scores)["ci_upper"],
        )
        result["open_n"] = len(open_judge_results)

    return result


# ---------------------------------------------------------------------------
# Experiment 2: Raw Features vs. Model Embeddings
# ---------------------------------------------------------------------------

def run_exp2_features_vs_embeddings(
    songs: List[SongRecord],
    llm: BaseModelWrapper,
    judge: Optional[LLMJudge] = None,
    *,
    mert_model: Optional[MERTWrapper] = None,
    clap_model_name: str = "laion/larger_clap_music",
) -> Dict[str, Any]:
    """Compare reasoning quality from raw features vs CLAP vs MERT embeddings.

    For each song:
      A) Generate reasoning from raw features (feature descriptions)
      B) Generate reasoning from CLAP embeddings (stringified)
      C) Generate reasoning from MERT embeddings (stringified)

    Then score each reasoning set via the judge.

    Parameters
    ----------
    songs : list of SongRecord
        Songs with ``features`` and ``audio_path``.
    llm : BaseModelWrapper
        LLM to generate reasoning (typically ClaudeTextWrapper).
    judge : LLMJudge, optional
        Judge for scoring.  If None, only generations are returned.
    mert_model : MERTWrapper, optional
        Pre-loaded MERT wrapper.  Loaded on demand if None.
    clap_model_name : str
        HuggingFace identifier for the CLAP model.

    Returns
    -------
    dict
        ``{"raw_features": {...}, "clap": {...}, "mert": {...},
           "comparison": {...}}``
    """
    results: Dict[str, Any] = {
        "raw_features": {"generations": [], "scores": []},
        "clap": {"generations": [], "scores": []},
        "mert": {"generations": [], "scores": []},
    }

    # Load CLAP if available
    clap_fn = _get_clap_embedder(clap_model_name)

    # Load MERT
    if mert_model is None:
        try:
            mert_model = MERTWrapper()
        except Exception as exc:
            logger.warning("Could not load MERT: %s", exc)
            mert_model = None

    question_template = (
        "Based on the following representation of a piece of music, "
        "explain why this music produces its emotional effect. "
        "What specific sonic elements drive the listener's response?\n\n{representation}"
    )

    for i, song in enumerate(songs):
        audio_path = song.get("audio_path", "")
        features = song.get("features", {})
        ref_reasoning = song.get("reference_reasoning", "")

        # --- A: Raw features ---
        from musicwhy.benchmark.question_generator import _describe_features
        feat_desc = _describe_features(features)
        prompt_a = question_template.format(representation=f"Audio features: {feat_desc}")
        gen_a = llm.answer_open(feat_desc, prompt_a)
        results["raw_features"]["generations"].append(gen_a)

        # --- B: CLAP embedding ---
        if clap_fn is not None and audio_path:
            try:
                clap_emb = clap_fn(audio_path)
                clap_str = f"CLAP embedding (768-dim): {clap_emb[:10].tolist()}... (truncated)"
                prompt_b = question_template.format(representation=clap_str)
                gen_b = llm.answer_open(clap_str, prompt_b)
            except Exception as exc:
                logger.warning("CLAP failed for song %d: %s", i, exc)
                gen_b = "[CLAP embedding extraction failed]"
        else:
            gen_b = "[CLAP not available]"
        results["clap"]["generations"].append(gen_b)

        # --- C: MERT embedding ---
        if mert_model is not None and audio_path:
            try:
                mert_emb = mert_model.get_embedding(audio_path)
                mert_str = f"MERT embedding (1024-dim): {mert_emb[:10].tolist()}... (truncated)"
                prompt_c = question_template.format(representation=mert_str)
                gen_c = llm.answer_open(mert_str, prompt_c)
            except Exception as exc:
                logger.warning("MERT failed for song %d: %s", i, exc)
                gen_c = "[MERT embedding extraction failed]"
        else:
            gen_c = "[MERT not available]"
        results["mert"]["generations"].append(gen_c)

        # --- Score if judge available ---
        if judge and ref_reasoning:
            for key, gen in [("raw_features", gen_a), ("clap", gen_b), ("mert", gen_c)]:
                try:
                    score = judge.evaluate_explanation(gen, ref_reasoning)
                    results[key]["scores"].append(score)
                except Exception as exc:
                    logger.warning("Judge failed for %s, song %d: %s", key, i, exc)

        if (i + 1) % 10 == 0:
            logger.info("Exp 2 progress: %d / %d songs", i + 1, len(songs))

    # Summarize
    results["comparison"] = {}
    for key in ("raw_features", "clap", "mert"):
        scores = [
            s.get("overall", s.get("score", 0))
            for s in results[key]["scores"]
        ]
        if scores:
            results["comparison"][key] = {
                "mean_score": float(np.mean(scores)),
                "ci": bootstrap_ci(scores),
                "n": len(scores),
            }

    return results


def _get_clap_embedder(model_name: str):
    """Try to load CLAP and return an embedding function, or None."""
    try:
        from transformers import ClapModel, ClapProcessor
        import librosa

        clap = ClapModel.from_pretrained(model_name)
        clap_proc = ClapProcessor.from_pretrained(model_name)

        def embed(audio_path: str) -> np.ndarray:
            y, sr = librosa.load(audio_path, sr=48000, mono=True)
            inputs = clap_proc(audios=[y], return_tensors="pt", sampling_rate=48000)
            features = clap.get_audio_features(**inputs)
            return features.detach().cpu().numpy().squeeze()

        return embed
    except Exception as exc:
        logger.warning("CLAP not available: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Experiment 3: Source-Aware Emotion Ablation
# ---------------------------------------------------------------------------

def run_exp3_source_emotion_ablation(
    songs_with_stems: List[SongRecord],
    models: List[BaseModelWrapper],
    judge: Optional[LLMJudge] = None,
) -> Dict[str, Any]:
    """Test whether models can identify which stem was removed from emotional change.

    For each song with stem separations, create ablated mixes (full mix minus
    one stem), then ask models which stem was removed based on the emotional
    difference.

    Parameters
    ----------
    songs_with_stems : list of SongRecord
        Each must have ``stem_features``, ``stem_paths`` (stem_name -> path),
        and optionally ``ablated_paths`` (stem_name -> path of mix without
        that stem).
    models : list of BaseModelWrapper
    judge : LLMJudge, optional

    Returns
    -------
    dict
        ``{model_name: {"accuracy": float, "per_stem": {...}, "ci": ...}}``
    """
    results: Dict[str, Any] = {}

    for model in models:
        model_name = model.name
        logger.info("Exp 3 - Evaluating: %s", model_name)

        preds: List[str] = []
        truths: List[str] = []
        per_stem: Dict[str, Dict[str, int]] = {}

        for song in songs_with_stems:
            stem_names = sorted(song.get("stem_features", {}).keys())
            ablated_paths = song.get("ablated_paths", {})
            full_path = song.get("audio_path", "")

            if not ablated_paths:
                continue

            for removed_stem in stem_names:
                ablated_audio = ablated_paths.get(removed_stem)
                if ablated_audio is None:
                    continue

                prompt = (
                    f"Listen to this audio clip. It is a song with one instrument/stem "
                    f"removed from the original mix. The possible stems are: "
                    f"{', '.join(stem_names)}.\n\n"
                    f"Which stem has been removed? What emotional quality is missing? "
                    f"Respond with the stem name first, then explain."
                )

                try:
                    response = model.answer_open(ablated_audio, prompt)
                    predicted = _extract_stem_name(response, stem_names)
                except NotImplementedError:
                    predicted = "unknown"
                except Exception as exc:
                    logger.warning("Prediction failed: %s", exc)
                    predicted = "unknown"

                preds.append(predicted)
                truths.append(removed_stem)

                per_stem.setdefault(removed_stem, {"correct": 0, "total": 0})
                per_stem[removed_stem]["total"] += 1
                if predicted == removed_stem:
                    per_stem[removed_stem]["correct"] += 1

        # Compute metrics
        if preds:
            acc = mc_accuracy(preds, truths)
            binary = [1.0 if p == g else 0.0 for p, g in zip(preds, truths)]
            ci = bootstrap_ci(binary)

            stem_accuracies = {
                stem: info["correct"] / info["total"] if info["total"] > 0 else 0.0
                for stem, info in per_stem.items()
            }
        else:
            acc = 0.0
            ci = bootstrap_ci([])
            stem_accuracies = {}

        results[model_name] = {
            "accuracy": acc,
            "n_predictions": len(preds),
            "per_stem_accuracy": stem_accuracies,
            "ci": (ci["ci_lower"], ci["ci_upper"]),
            "baseline_random": 1.0 / max(len(per_stem), 1),
        }

    return results


def _extract_stem_name(response: str, valid_stems: List[str]) -> str:
    """Extract a stem name from a free-text model response."""
    response_lower = response.lower()
    for stem in valid_stems:
        if stem.lower() in response_lower:
            return stem
    # Fallback: first word
    first_word = response.split()[0].strip(".,!?:;").lower() if response.strip() else ""
    for stem in valid_stems:
        if stem.lower().startswith(first_word):
            return stem
    return valid_stems[0] if valid_stems else "unknown"


# ---------------------------------------------------------------------------
# Experiment 4: Sound-World Correlation
# ---------------------------------------------------------------------------

def run_exp4_sound_world_correlation(
    chart_data: Any,
    economic_data: Any,
    *,
    feature_columns: Optional[List[str]] = None,
    economic_columns: Optional[List[str]] = None,
    max_lag: int = 5,
    year_range: Tuple[int, int] = (1960, 2024),
) -> Dict[str, Any]:
    """Analyze correlations between audio features of popular music and economic indicators.

    Parameters
    ----------
    chart_data : DataFrame-like
        Must have columns ``year`` and audio feature columns
        (``avg_tempo``, ``avg_brightness``, etc.).
    economic_data : DataFrame-like
        Must have columns ``year`` and economic indicators
        (``gdp_growth``, ``unemployment``, etc.).
    feature_columns : list of str, optional
        Audio feature column names.  Defaults to
        ``["avg_tempo", "avg_brightness", "avg_energy", "avg_valence"]``.
    economic_columns : list of str, optional
        Economic indicator column names.  Defaults to
        ``["gdp_growth", "unemployment"]``.
    max_lag : int
        Maximum lag for correlation analysis.
    year_range : tuple of int
        (start_year, end_year) inclusive.

    Returns
    -------
    dict
        ``{"correlations": {(feature, indicator): {lag: {r, p, n}}},
           "summary": {...}}``
    """
    import pandas as pd

    if feature_columns is None:
        feature_columns = ["avg_tempo", "avg_brightness", "avg_energy", "avg_valence"]
    if economic_columns is None:
        economic_columns = ["gdp_growth", "unemployment"]

    # Convert to DataFrames if not already
    chart_df = pd.DataFrame(chart_data) if not isinstance(chart_data, pd.DataFrame) else chart_data
    econ_df = pd.DataFrame(economic_data) if not isinstance(economic_data, pd.DataFrame) else economic_data

    # Filter year range
    chart_df = chart_df[
        (chart_df["year"] >= year_range[0]) & (chart_df["year"] <= year_range[1])
    ].sort_values("year")
    econ_df = econ_df[
        (econ_df["year"] >= year_range[0]) & (econ_df["year"] <= year_range[1])
    ].sort_values("year")

    # Merge on year
    merged = pd.merge(chart_df, econ_df, on="year", how="inner")

    correlations: Dict[str, Dict[int, Dict[str, float]]] = {}
    summary: Dict[str, Any] = {}

    for feat_col in feature_columns:
        if feat_col not in merged.columns:
            logger.warning("Feature column '%s' not found; skipping.", feat_col)
            continue

        for econ_col in economic_columns:
            if econ_col not in merged.columns:
                logger.warning("Economic column '%s' not found; skipping.", econ_col)
                continue

            pair_key = f"{feat_col}_vs_{econ_col}"
            feat_series = merged[feat_col].dropna().values
            econ_series = merged[econ_col].dropna().values

            # Align lengths
            n = min(len(feat_series), len(econ_series))
            if n < 5:
                logger.warning(
                    "Too few data points for %s (%d); skipping.", pair_key, n,
                )
                continue

            lag_results = pearson_with_lag(
                feat_series[:n], econ_series[:n], max_lag=max_lag,
            )
            correlations[pair_key] = lag_results

            # Find the lag with the strongest correlation
            best_lag = max(
                lag_results.keys(),
                key=lambda l: abs(lag_results[l].get("r", 0))
                if not np.isnan(lag_results[l].get("r", float("nan")))
                else 0,
            )
            best_r = lag_results[best_lag]

            summary[pair_key] = {
                "best_lag": best_lag,
                "best_r": best_r["r"],
                "best_p": best_r["p"],
                "significant": best_r["p"] < 0.05 if not np.isnan(best_r["p"]) else False,
                "n_years": n,
                "zero_lag_r": lag_results.get(0, {}).get("r", float("nan")),
                "zero_lag_p": lag_results.get(0, {}).get("p", float("nan")),
            }

    return {
        "correlations": correlations,
        "summary": summary,
        "year_range": year_range,
        "n_merged_years": len(merged) if "merged" in dir() else 0,
    }


# ---------------------------------------------------------------------------
# Experiment 5: Instrumental vs. Vocal Performance Gap
# ---------------------------------------------------------------------------

def run_exp5_instrumental_vs_vocal(
    models: List[BaseModelWrapper],
    benchmark: List[Question],
    judge: Optional[LLMJudge] = None,
) -> Dict[str, Any]:
    """Compare model performance on instrumental vs. vocal music questions.

    Questions must have ``metadata.has_vocals`` (bool) or
    ``metadata.category`` in (``"instrumental"``, ``"vocal"``).

    Parameters
    ----------
    models : list of BaseModelWrapper
    benchmark : list of Question
        Questions with vocal/instrumental metadata.
    judge : LLMJudge, optional
        Required for open-ended scoring.

    Returns
    -------
    dict
        ``{model_name: {"instrumental": {...}, "vocal": {...},
           "gap": {...}, "per_task_gap": {...}}}``
    """
    # Split benchmark into instrumental vs vocal
    instrumental_qs: List[Question] = []
    vocal_qs: List[Question] = []

    for q in benchmark:
        meta = q.get("metadata", {})
        has_vocals = meta.get("has_vocals")
        category = meta.get("category", "").lower()

        if has_vocals is False or category == "instrumental":
            instrumental_qs.append(q)
        elif has_vocals is True or category == "vocal":
            vocal_qs.append(q)
        # Ignore questions without vocal/instrumental annotation

    if not instrumental_qs or not vocal_qs:
        logger.warning(
            "Insufficient labeled data: %d instrumental, %d vocal questions.",
            len(instrumental_qs), len(vocal_qs),
        )
        return {"error": "Insufficient vocal/instrumental labeled questions."}

    results: Dict[str, Any] = {}

    for model in models:
        model_name = model.name
        logger.info("Exp 5 - Evaluating: %s", model_name)

        inst_scores = _collect_scores(model, instrumental_qs, judge)
        vocal_scores = _collect_scores(model, vocal_qs, judge)

        gap_result = performance_gap(vocal_scores, inst_scores)

        # Per-task breakdown
        per_task_gap: Dict[int, Dict[str, Any]] = {}
        task_ids = set(q.get("task_id", 0) for q in benchmark)
        for tid in sorted(task_ids):
            inst_t = [q for q in instrumental_qs if q.get("task_id") == tid]
            vocal_t = [q for q in vocal_qs if q.get("task_id") == tid]
            if inst_t and vocal_t:
                inst_s = _collect_scores(model, inst_t, judge)
                vocal_s = _collect_scores(model, vocal_t, judge)
                if inst_s and vocal_s:
                    per_task_gap[tid] = performance_gap(vocal_s, inst_s)

        results[model_name] = {
            "instrumental": {
                "n": len(inst_scores),
                "mean": float(np.mean(inst_scores)) if inst_scores else 0.0,
                "ci": bootstrap_ci(inst_scores),
            },
            "vocal": {
                "n": len(vocal_scores),
                "mean": float(np.mean(vocal_scores)) if vocal_scores else 0.0,
                "ci": bootstrap_ci(vocal_scores),
            },
            "gap": gap_result,
            "per_task_gap": per_task_gap,
        }

    return results


def _collect_scores(
    model: BaseModelWrapper,
    questions: List[Question],
    judge: Optional[LLMJudge],
) -> List[float]:
    """Collect numeric scores for a set of questions from a model.

    For MC questions: 1.0 for correct, 0.0 for wrong.
    For open-ended: judge score (1-5), normalized to [0, 1].
    """
    scores: List[float] = []

    for q in questions:
        audio = q.get("audio_reference", "")
        q_text = q.get("question_text", "")
        options = q.get("options")
        correct = q.get("correct_answer")
        ref_expl = q.get("reference_explanation", "")

        try:
            if options and correct:
                pred = model.answer_mc(audio, q_text, options)
                scores.append(1.0 if str(pred).strip().upper() == str(correct).strip().upper() else 0.0)
            elif ref_expl and judge:
                response = model.answer_open(audio, q_text)
                result = judge.evaluate_open_ended(response, ref_expl)
                # Normalize 1-5 score to 0-1
                scores.append((result.get("score", 1) - 1) / 4.0)
        except NotImplementedError:
            continue
        except Exception as exc:
            logger.warning("Score collection failed: %s", exc)
            continue

    return scores
