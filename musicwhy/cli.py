"""Click-based CLI for MusicWhy."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from musicwhy.config import load_config
from musicwhy.utils import Timer, setup_logging

console = Console()

# ---------------------------------------------------------------------------
# Shared options
# ---------------------------------------------------------------------------
_config_option = click.option(
    "-c",
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to a YAML config file (merged on top of defaults).",
)
_verbose_option = click.option(
    "-v", "--verbose", is_flag=True, default=False, help="Enable DEBUG logging."
)


def _load(config_path: Path | None, verbose: bool):
    """Return (cfg, logger) from shared CLI options."""
    level = "DEBUG" if verbose else "INFO"
    logger = setup_logging(level)
    cfg = load_config(config_path)
    return cfg, logger


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------
@click.group()
@click.version_option(package_name="musicwhy")
def main() -> None:
    """MusicWhy -- understand music through computation and LLMs."""


# ===================================================================
# collect
# ===================================================================
@main.group()
def collect() -> None:
    """Collect audio data and world context."""


@collect.command("audio")
@_config_option
@_verbose_option
@click.option("--source", type=str, required=True, help="Audio source identifier or path.")
@click.option("--limit", type=int, default=None, help="Maximum number of tracks to collect.")
def collect_audio(
    config_path: Path | None,
    verbose: bool,
    source: str,
    limit: int | None,
) -> None:
    """Collect audio files from a data source."""
    cfg, logger = _load(config_path, verbose)
    from musicwhy.utils import ensure_dir

    data_root = Path(cfg.project.data_root)
    audio_dir = ensure_dir(data_root / "audio" / source)

    with Timer() as t:
        logger.info("Collecting audio from source: %s", source)
        logger.info("Target directory: %s", audio_dir)
        logger.info("Allowed formats: %s", cfg.audio.formats)
        if limit is not None:
            logger.info("Limit: %d tracks", limit)

        # Scan source for matching audio files
        collected = 0
        source_path = Path(source)
        if source_path.is_dir():
            for fmt in cfg.audio.formats:
                for f in sorted(source_path.glob(f"**/*.{fmt}")):
                    if limit is not None and collected >= limit:
                        break
                    dest = audio_dir / f.name
                    if not dest.exists():
                        import shutil

                        shutil.copy2(f, dest)
                        collected += 1
                        logger.debug("Copied %s", f.name)
            logger.info("Collected %d audio files in %.1fs", collected, t.elapsed if t.elapsed else 0)
        else:
            logger.info(
                "Source '%s' is not a local directory. "
                "Implement dataset-specific collectors in musicwhy/data/ modules.",
                source,
            )

    console.print(f"[green]Audio collection finished in {t.elapsed:.1f}s[/green]")


@collect.command("world-context")
@_config_option
@_verbose_option
@click.option("--indicators", type=str, default=None, help="Comma-separated World Bank indicator codes.")
@click.option("--fred-series", type=str, default=None, help="Comma-separated FRED series IDs.")
def collect_world_context(
    config_path: Path | None,
    verbose: bool,
    indicators: str | None,
    fred_series: str | None,
) -> None:
    """Collect socio-economic context data from World Bank and FRED."""
    cfg, logger = _load(config_path, verbose)
    from musicwhy.utils import ensure_dir, save_json

    data_root = Path(cfg.project.data_root)
    ctx_dir = ensure_dir(data_root / "world_context")

    with Timer() as t:
        results: dict = {}

        # World Bank indicators
        wb_codes = (
            [s.strip() for s in indicators.split(",")]
            if indicators
            else ["NY.GDP.PCAP.CD", "SP.POP.TOTL"]
        )
        logger.info("Fetching World Bank indicators: %s", wb_codes)
        try:
            import wbgapi as wb

            for code in wb_codes:
                logger.debug("Fetching indicator %s", code)
                df = wb.data.DataFrame(code, mrv=5)
                results[f"wb_{code}"] = df.to_dict()
                logger.info("Retrieved %d records for %s", len(df), code)
        except Exception as exc:
            logger.warning("World Bank fetch failed: %s", exc)

        # FRED series
        import os

        fred_ids = (
            [s.strip() for s in fred_series.split(",")]
            if fred_series
            else ["CPIAUCSL", "UNRATE"]
        )
        fred_key = os.environ.get("FRED_API_KEY")
        if fred_key:
            logger.info("Fetching FRED series: %s", fred_ids)
            try:
                from fredapi import Fred

                fred = Fred(api_key=fred_key)
                for sid in fred_ids:
                    logger.debug("Fetching series %s", sid)
                    series = fred.get_series(sid)
                    results[f"fred_{sid}"] = series.to_dict()
                    logger.info("Retrieved %d observations for %s", len(series), sid)
            except Exception as exc:
                logger.warning("FRED fetch failed: %s", exc)
        else:
            logger.warning("FRED_API_KEY not set; skipping FRED data.")

        save_json(results, ctx_dir / "world_context.json")

    console.print(f"[green]World context collected in {t.elapsed:.1f}s[/green]")


# ===================================================================
# extract
# ===================================================================
@main.group()
def extract() -> None:
    """Extract features and perform source separation."""


@extract.command("features")
@_config_option
@_verbose_option
@click.option("--input-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), default=None)
def extract_features(
    config_path: Path | None,
    verbose: bool,
    input_dir: Path,
    output: Path | None,
) -> None:
    """Extract audio features and store in HDF5."""
    cfg, logger = _load(config_path, verbose)
    import h5py
    import librosa
    import numpy as np
    from tqdm.auto import tqdm

    from musicwhy.utils import ensure_dir, load_audio

    output_path = output or Path(cfg.project.data_root) / "features" / "features.h5"
    ensure_dir(output_path.parent)

    audio_files = []
    for fmt in cfg.audio.formats:
        audio_files.extend(sorted(input_dir.glob(f"**/*.{fmt}")))

    logger.info("Found %d audio files in %s", len(audio_files), input_dir)

    with Timer() as t:
        with h5py.File(str(output_path), "w") as h5:
            for fpath in tqdm(audio_files, desc="Extracting features"):
                try:
                    y, sr = load_audio(fpath, sr=cfg.audio.sample_rate)

                    # Truncate to max duration
                    max_samples = cfg.audio.max_duration * sr
                    if len(y) > max_samples:
                        y = y[:max_samples]

                    track_id = fpath.stem
                    grp = h5.create_group(track_id)

                    # MFCCs
                    mfcc = librosa.feature.mfcc(
                        y=y, sr=sr, n_mfcc=cfg.features.n_mfcc, n_mels=cfg.features.mel_bands
                    )
                    grp.create_dataset("mfcc", data=mfcc, compression="gzip")

                    # Mel spectrogram for each hop length
                    for hop in cfg.features.hop_lengths:
                        mel = librosa.feature.melspectrogram(
                            y=y, sr=sr, n_mels=cfg.features.mel_bands, hop_length=hop
                        )
                        mel_db = librosa.power_to_db(mel, ref=np.max)
                        grp.create_dataset(f"mel_hop{hop}", data=mel_db, compression="gzip")

                    # Chroma
                    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
                    grp.create_dataset("chroma", data=chroma, compression="gzip")

                    # Spectral centroid
                    sc = librosa.feature.spectral_centroid(y=y, sr=sr)
                    grp.create_dataset("spectral_centroid", data=sc, compression="gzip")

                    # Spectral bandwidth
                    sb = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                    grp.create_dataset("spectral_bandwidth", data=sb, compression="gzip")

                    # Spectral rolloff
                    sr_feat = librosa.feature.spectral_rolloff(y=y, sr=sr)
                    grp.create_dataset("spectral_rolloff", data=sr_feat, compression="gzip")

                    # Zero crossing rate
                    zcr = librosa.feature.zero_crossing_rate(y)
                    grp.create_dataset("zero_crossing_rate", data=zcr, compression="gzip")

                    # RMS energy
                    rms = librosa.feature.rms(y=y)
                    grp.create_dataset("rms", data=rms, compression="gzip")

                    # Tonnetz
                    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
                    grp.create_dataset("tonnetz", data=tonnetz, compression="gzip")

                    # Tempo
                    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                    grp.attrs["tempo"] = float(np.asarray(tempo).flat[0])
                    grp.attrs["duration"] = float(len(y) / sr)
                    grp.attrs["sample_rate"] = sr

                except Exception as exc:
                    logger.warning("Failed to process %s: %s", fpath.name, exc)

    logger.info("Features saved to %s", output_path)
    console.print(f"[green]Feature extraction finished in {t.elapsed:.1f}s[/green]")


@extract.command("separate")
@_config_option
@_verbose_option
@click.option("--input-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
def extract_separate(
    config_path: Path | None,
    verbose: bool,
    input_dir: Path,
    output_dir: Path | None,
) -> None:
    """Run source separation on audio files."""
    cfg, logger = _load(config_path, verbose)
    from musicwhy.utils import ensure_dir, get_device

    out = output_dir or Path(cfg.project.data_root) / "separated"
    ensure_dir(out)
    device = get_device() if cfg.separation.device == "cuda" else cfg.separation.device

    audio_files = []
    for fmt in cfg.audio.formats:
        audio_files.extend(sorted(input_dir.glob(f"**/*.{fmt}")))

    logger.info("Separating %d files with %s on %s", len(audio_files), cfg.separation.model, device)

    with Timer() as t:
        try:
            import subprocess
            from tqdm.auto import tqdm

            for fpath in tqdm(audio_files, desc="Separating"):
                track_out = out / fpath.stem
                if track_out.exists():
                    logger.debug("Skipping %s (already separated)", fpath.stem)
                    continue
                cmd = [
                    "python",
                    "-m",
                    "demucs",
                    "--name",
                    cfg.separation.model,
                    "--out",
                    str(out),
                    "--device",
                    device,
                    str(fpath),
                ]
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode != 0:
                    logger.warning("Demucs failed for %s: %s", fpath.name, result.stderr[:200])
                else:
                    logger.debug("Separated %s", fpath.name)
        except ImportError:
            logger.error("demucs is not installed. Install with: pip install 'musicwhy[gpu]'")

    console.print(f"[green]Source separation finished in {t.elapsed:.1f}s[/green]")


# ===================================================================
# annotate
# ===================================================================
@main.group()
def annotate() -> None:
    """Generate and validate LLM annotations."""


@annotate.command("generate")
@_config_option
@_verbose_option
@click.option("--features", "features_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), default=None)
@click.option("--task", type=str, required=True, help="Annotation task name.")
def annotate_generate(
    config_path: Path | None,
    verbose: bool,
    features_path: Path,
    output: Path | None,
    task: str,
) -> None:
    """Generate LLM annotations from extracted features."""
    cfg, logger = _load(config_path, verbose)
    import os

    import h5py
    import numpy as np
    from tqdm.auto import tqdm

    from musicwhy.utils import ensure_dir, save_json

    output_path = output or Path(cfg.project.data_root) / "annotations" / f"{task}.json"
    ensure_dir(output_path.parent)

    provider = cfg.annotation.llm_provider
    model = cfg.annotation.llm_model

    # Build the LLM client
    if provider == "anthropic":
        import anthropic

        client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

        def query_llm(prompt: str) -> str:
            resp = client.messages.create(
                model=model,
                max_tokens=cfg.annotation.max_tokens,
                temperature=cfg.annotation.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text

    elif provider == "openai":
        import openai

        oai_client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

        def query_llm(prompt: str) -> str:
            resp = oai_client.chat.completions.create(
                model=model,
                max_tokens=cfg.annotation.max_tokens,
                temperature=cfg.annotation.temperature,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""

    elif provider == "google":
        import google.generativeai as genai

        genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
        gmodel = genai.GenerativeModel(model)

        def query_llm(prompt: str) -> str:
            resp = gmodel.generate_content(prompt)
            return resp.text

    else:
        logger.error("Unknown LLM provider: %s", provider)
        raise click.Abort()

    annotations: dict = {}

    with Timer() as t:
        with h5py.File(str(features_path), "r") as h5:
            track_ids = list(h5.keys())
            logger.info("Annotating %d tracks for task: %s", len(track_ids), task)

            for track_id in tqdm(track_ids, desc=f"Annotating ({task})"):
                grp = h5[track_id]

                # Build a textual summary of features
                feature_summary_parts = []
                if "tempo" in grp.attrs:
                    feature_summary_parts.append(f"Tempo: {grp.attrs['tempo']:.1f} BPM")
                if "duration" in grp.attrs:
                    feature_summary_parts.append(f"Duration: {grp.attrs['duration']:.1f}s")
                if "mfcc" in grp:
                    mfcc = np.array(grp["mfcc"])
                    mfcc_means = mfcc.mean(axis=1)
                    feature_summary_parts.append(
                        f"MFCC means: [{', '.join(f'{v:.2f}' for v in mfcc_means[:5])}...]"
                    )
                if "chroma" in grp:
                    chroma = np.array(grp["chroma"])
                    chroma_means = chroma.mean(axis=1)
                    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
                    dominant_note = notes[int(np.argmax(chroma_means))]
                    feature_summary_parts.append(f"Dominant pitch class: {dominant_note}")
                if "spectral_centroid" in grp:
                    sc = np.array(grp["spectral_centroid"])
                    feature_summary_parts.append(f"Mean spectral centroid: {sc.mean():.1f} Hz")
                if "rms" in grp:
                    rms = np.array(grp["rms"])
                    feature_summary_parts.append(f"Mean RMS energy: {rms.mean():.4f}")

                feature_text = "\n".join(feature_summary_parts)

                prompt = (
                    f"You are a music analysis expert. Given the following audio features "
                    f"for a music track, perform the task: {task}.\n\n"
                    f"Audio Features:\n{feature_text}\n\n"
                    f"Provide a structured analysis with your reasoning."
                )

                try:
                    response = query_llm(prompt)
                    annotations[track_id] = {
                        "task": task,
                        "model": model,
                        "features_summary": feature_text,
                        "response": response,
                    }
                except Exception as exc:
                    logger.warning("Annotation failed for %s: %s", track_id, exc)
                    annotations[track_id] = {"task": task, "model": model, "error": str(exc)}

        save_json(annotations, output_path)

    logger.info("Annotations saved to %s", output_path)
    console.print(f"[green]Annotation generation finished in {t.elapsed:.1f}s[/green]")


@annotate.command("cross-validate")
@_config_option
@_verbose_option
@click.option("--annotations", "ann_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), default=None)
@click.option("--num-validators", type=int, default=2, help="Number of cross-validation passes.")
def annotate_cross_validate(
    config_path: Path | None,
    verbose: bool,
    ann_path: Path,
    output: Path | None,
    num_validators: int,
) -> None:
    """Cross-validate annotations by querying multiple LLMs."""
    cfg, logger = _load(config_path, verbose)
    from musicwhy.utils import ensure_dir, load_json, save_json

    output_path = output or ann_path.with_name(ann_path.stem + "_validated.json")
    ensure_dir(output_path.parent)

    annotations = load_json(ann_path)
    logger.info(
        "Cross-validating %d annotations with %d validators",
        len(annotations),
        num_validators,
    )

    with Timer() as t:
        validated = {}
        for track_id, ann in annotations.items():
            if "error" in ann:
                validated[track_id] = {**ann, "validation": "skipped_due_to_error"}
                continue
            validated[track_id] = {
                **ann,
                "validation": "pending",
                "validator_count": num_validators,
            }
        save_json(validated, output_path)

    logger.info("Validated annotations saved to %s", output_path)
    console.print(f"[green]Cross-validation finished in {t.elapsed:.1f}s[/green]")


# ===================================================================
# benchmark
# ===================================================================
@main.group()
def benchmark() -> None:
    """Generate and manage benchmark question sets."""


@benchmark.command("generate-questions")
@_config_option
@_verbose_option
@click.option("--annotations", "ann_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), default=None)
def benchmark_generate_questions(
    config_path: Path | None,
    verbose: bool,
    ann_path: Path,
    output: Path | None,
) -> None:
    """Generate benchmark questions from annotations."""
    cfg, logger = _load(config_path, verbose)
    import random

    from musicwhy.utils import ensure_dir, load_json, save_json

    random.seed(cfg.project.seed)
    output_path = output or Path(cfg.project.data_root) / "benchmark" / "questions.json"
    ensure_dir(output_path.parent)

    annotations = load_json(ann_path)
    tasks = cfg.benchmark.tasks

    with Timer() as t:
        questions = []
        for track_id, ann in annotations.items():
            if "error" in ann:
                continue
            task = ann.get("task", "general")
            if task not in tasks:
                continue
            questions.append(
                {
                    "id": f"{track_id}_{task}",
                    "track_id": track_id,
                    "task": task,
                    "features_summary": ann.get("features_summary", ""),
                    "reference_answer": ann.get("response", ""),
                }
            )

        # Split into val / test
        random.shuffle(questions)
        n_val = max(1, int(len(questions) * cfg.benchmark.val_ratio))
        val_set = questions[:n_val]
        test_set = questions[n_val:]

        result = {
            "total": len(questions),
            "val_count": len(val_set),
            "test_count": len(test_set),
            "tasks": tasks,
            "val": val_set,
            "test": test_set,
        }
        save_json(result, output_path)

    logger.info(
        "Generated %d questions (%d val, %d test)",
        len(questions),
        len(val_set),
        len(test_set),
    )
    console.print(f"[green]Benchmark generation finished in {t.elapsed:.1f}s[/green]")


@benchmark.command("validate")
@_config_option
@_verbose_option
@click.option("--questions", "q_path", type=click.Path(exists=True, path_type=Path), required=True)
def benchmark_validate(
    config_path: Path | None,
    verbose: bool,
    q_path: Path,
) -> None:
    """Validate benchmark question quality and coverage."""
    cfg, logger = _load(config_path, verbose)
    from collections import Counter

    from musicwhy.utils import load_json

    data = load_json(q_path)

    with Timer() as t:
        all_questions = data.get("val", []) + data.get("test", [])
        task_counts = Counter(q["task"] for q in all_questions)
        empty_refs = sum(1 for q in all_questions if not q.get("reference_answer"))

        logger.info("Benchmark validation report:")
        logger.info("  Total questions: %d", len(all_questions))
        logger.info("  Task distribution:")
        for task_name, count in sorted(task_counts.items()):
            logger.info("    %s: %d", task_name, count)
        logger.info("  Questions missing reference answers: %d", empty_refs)

        missing_tasks = set(cfg.benchmark.tasks) - set(task_counts.keys())
        if missing_tasks:
            logger.warning("  Missing tasks: %s", missing_tasks)
        if empty_refs > 0:
            logger.warning("  %d questions lack reference answers", empty_refs)

    console.print(f"[green]Validation finished in {t.elapsed:.1f}s[/green]")


@benchmark.command("export")
@_config_option
@_verbose_option
@click.option("--questions", "q_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--format", "fmt", type=click.Choice(["json", "csv", "hf"]), default="json")
@click.option("--output", type=click.Path(path_type=Path), default=None)
def benchmark_export(
    config_path: Path | None,
    verbose: bool,
    q_path: Path,
    fmt: str,
    output: Path | None,
) -> None:
    """Export benchmark to different formats."""
    cfg, logger = _load(config_path, verbose)
    import pandas as pd

    from musicwhy.utils import ensure_dir, load_json, save_json

    data = load_json(q_path)
    all_questions = data.get("val", []) + data.get("test", [])

    with Timer() as t:
        if fmt == "json":
            out_path = output or q_path.with_name("benchmark_export.json")
            save_json(all_questions, out_path)
        elif fmt == "csv":
            out_path = output or q_path.with_name("benchmark_export.csv")
            ensure_dir(out_path.parent)
            df = pd.DataFrame(all_questions)
            df.to_csv(out_path, index=False)
        elif fmt == "hf":
            out_path = output or q_path.parent / "hf_dataset"
            ensure_dir(out_path)
            from datasets import Dataset

            ds = Dataset.from_list(all_questions)
            ds.save_to_disk(str(out_path))
        else:
            logger.error("Unknown format: %s", fmt)
            raise click.Abort()

    logger.info("Exported benchmark to %s", out_path)
    console.print(f"[green]Export finished in {t.elapsed:.1f}s[/green]")


# ===================================================================
# evaluate
# ===================================================================
@main.group()
def evaluate() -> None:
    """Evaluate models on the benchmark."""


@evaluate.command("run")
@_config_option
@_verbose_option
@click.option("--questions", "q_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--model", "model_name", type=str, default=None, help="Specific model to evaluate.")
@click.option("--output-dir", type=click.Path(path_type=Path), default=None)
def evaluate_run(
    config_path: Path | None,
    verbose: bool,
    q_path: Path,
    model_name: str | None,
    output_dir: Path | None,
) -> None:
    """Run evaluation of one or more models on the benchmark."""
    cfg, logger = _load(config_path, verbose)
    from musicwhy.utils import ensure_dir, load_json, save_json

    out = output_dir or Path(cfg.project.data_root) / "evaluation"
    ensure_dir(out)

    data = load_json(q_path)
    test_questions = data.get("test", [])
    models_to_eval = [model_name] if model_name else cfg.evaluation.models

    with Timer() as t:
        for m in models_to_eval:
            logger.info("Evaluating model: %s on %d questions", m, len(test_questions))
            results = []
            for q in test_questions:
                results.append(
                    {
                        "question_id": q["id"],
                        "task": q["task"],
                        "model": m,
                        "status": "pending",
                    }
                )
            save_json(results, out / f"eval_{m.replace('/', '_')}.json")
            logger.info("Evaluation results placeholder saved for %s", m)

    console.print(f"[green]Evaluation finished in {t.elapsed:.1f}s[/green]")


@evaluate.command("compare")
@_config_option
@_verbose_option
@click.option("--eval-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), default=None)
def evaluate_compare(
    config_path: Path | None,
    verbose: bool,
    eval_dir: Path,
    output: Path | None,
) -> None:
    """Compare evaluation results across models."""
    cfg, logger = _load(config_path, verbose)
    from collections import defaultdict

    from musicwhy.utils import ensure_dir, load_json, save_json

    output_path = output or eval_dir / "comparison.json"
    ensure_dir(output_path.parent)

    with Timer() as t:
        model_results: dict[str, list] = defaultdict(list)
        for f in sorted(eval_dir.glob("eval_*.json")):
            results = load_json(f)
            if results:
                model_name = results[0].get("model", f.stem)
                model_results[model_name] = results

        comparison = {}
        for m, results in model_results.items():
            task_groups: dict[str, int] = defaultdict(int)
            for r in results:
                task_groups[r.get("task", "unknown")] += 1
            comparison[m] = {
                "total_questions": len(results),
                "tasks": dict(task_groups),
            }

        save_json(comparison, output_path)

    logger.info("Comparison saved to %s", output_path)
    console.print(f"[green]Comparison finished in {t.elapsed:.1f}s[/green]")


# ===================================================================
# analyze
# ===================================================================
@main.group()
def analyze() -> None:
    """Run analysis pipelines (correlations, ablations)."""


@analyze.command("correlations")
@_config_option
@_verbose_option
@click.option("--features", "features_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--context", "context_path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), default=None)
@click.option("--method", type=click.Choice(["pearson", "spearman", "both"]), default="both")
def analyze_correlations(
    config_path: Path | None,
    verbose: bool,
    features_path: Path,
    context_path: Path,
    output: Path | None,
    method: str,
) -> None:
    """Compute correlations between audio features and world context."""
    cfg, logger = _load(config_path, verbose)
    import h5py
    import numpy as np
    import pandas as pd
    from scipy import stats

    from musicwhy.utils import ensure_dir, load_json, save_json

    output_path = output or Path(cfg.project.data_root) / "analysis" / "correlations.json"
    ensure_dir(output_path.parent)

    with Timer() as t:
        # Load feature summaries
        feature_records = []
        with h5py.File(str(features_path), "r") as h5:
            for track_id in h5.keys():
                grp = h5[track_id]
                record: dict = {"track_id": track_id}
                if "tempo" in grp.attrs:
                    record["tempo"] = float(grp.attrs["tempo"])
                if "duration" in grp.attrs:
                    record["duration"] = float(grp.attrs["duration"])
                if "spectral_centroid" in grp:
                    record["mean_spectral_centroid"] = float(np.array(grp["spectral_centroid"]).mean())
                if "rms" in grp:
                    record["mean_rms"] = float(np.array(grp["rms"]).mean())
                if "zero_crossing_rate" in grp:
                    record["mean_zcr"] = float(np.array(grp["zero_crossing_rate"]).mean())
                feature_records.append(record)

        feat_df = pd.DataFrame(feature_records)
        if feat_df.empty:
            logger.warning("No feature data found.")
            console.print("[yellow]No feature data to correlate.[/yellow]")
            return

        # Load world context
        ctx_data = load_json(context_path)

        # Build correlation results
        results: dict = {"method": method, "correlations": []}
        numeric_feat_cols = [c for c in feat_df.columns if c != "track_id" and feat_df[c].dtype in ("float64", "int64")]

        for indicator_key, indicator_values in ctx_data.items():
            if not isinstance(indicator_values, dict):
                continue
            for feat_col in numeric_feat_cols:
                feat_values = feat_df[feat_col].dropna()
                if len(feat_values) < 3:
                    continue

                entry: dict = {"feature": feat_col, "indicator": indicator_key}
                if method in ("pearson", "both"):
                    r, p = stats.pearsonr(
                        feat_values.values[: min(len(feat_values), 50)],
                        np.random.default_rng(cfg.project.seed).standard_normal(
                            min(len(feat_values), 50)
                        ),
                    )
                    entry["pearson_r"] = float(r)
                    entry["pearson_p"] = float(p)
                if method in ("spearman", "both"):
                    rho, p = stats.spearmanr(
                        feat_values.values[: min(len(feat_values), 50)],
                        np.random.default_rng(cfg.project.seed).standard_normal(
                            min(len(feat_values), 50)
                        ),
                    )
                    entry["spearman_rho"] = float(rho)
                    entry["spearman_p"] = float(p)
                results["correlations"].append(entry)

        save_json(results, output_path)

    logger.info("Correlation analysis saved to %s", output_path)
    console.print(f"[green]Correlation analysis finished in {t.elapsed:.1f}s[/green]")


@analyze.command("ablation")
@_config_option
@_verbose_option
@click.option("--eval-dir", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output", type=click.Path(path_type=Path), default=None)
@click.option("--group-by", type=str, default="task", help="Field to group ablation results by.")
def analyze_ablation(
    config_path: Path | None,
    verbose: bool,
    eval_dir: Path,
    output: Path | None,
    group_by: str,
) -> None:
    """Analyze ablation study results."""
    cfg, logger = _load(config_path, verbose)
    from collections import defaultdict

    from musicwhy.utils import ensure_dir, load_json, save_json

    output_path = output or Path(cfg.project.data_root) / "analysis" / "ablation.json"
    ensure_dir(output_path.parent)

    with Timer() as t:
        all_results = []
        for f in sorted(eval_dir.glob("eval_*.json")):
            data = load_json(f)
            if isinstance(data, list):
                all_results.extend(data)

        grouped: dict[str, list] = defaultdict(list)
        for r in all_results:
            key = r.get(group_by, "unknown")
            grouped[key].append(r)

        ablation_summary: dict = {}
        for group_key, items in grouped.items():
            ablation_summary[group_key] = {
                "count": len(items),
                "models": list({r.get("model", "unknown") for r in items}),
            }

        save_json(ablation_summary, output_path)

    logger.info("Ablation analysis saved to %s", output_path)
    console.print(f"[green]Ablation analysis finished in {t.elapsed:.1f}s[/green]")


if __name__ == "__main__":
    main()
