"""Phase 3: reasoning annotation for every manifest track with features.

Kept to light imports (csv, h5py, numpy, anthropic): pandas/librosa via TrackManifest and
FeatureExtractor took longer than a 2 h job to import on cold compute nodes reading from scratch.
Env: DATA_ROOT, SHARD, TRACKS_PER_SHARD, and CODE_ROOT (where the musicwhy package is imported
from; the job points it at a node-local copy). Existing annotation files are skipped (resume).
"""
import csv
import json
import os
import sys
from pathlib import Path

CODE_ROOT = Path(os.environ.get("CODE_ROOT") or os.environ["REPO"])
DATA = Path(os.environ["DATA_ROOT"])
SHARD = int(os.environ.get("SHARD", "0"))
TRACKS_PER_SHARD = int(os.environ.get("TRACKS_PER_SHARD", "140"))
STEMS = ["bass", "drums", "vocals", "other"]
MAX_CHART_ROWS = 10

sys.path.insert(0, str(CODE_ROOT))
print("Python started, importing modules", flush=True)

import h5py  # noqa: E402

from musicwhy.annotation.llm_reasoning import ReasoningConfig, ReasoningGenerator  # noqa: E402
from musicwhy.data.world_context_db import get_world_context  # noqa: E402
from musicwhy.features.feature_to_language import (  # noqa: E402
    features_to_description,
    stem_comparison_description,
)


def load_manifest():
    with open(DATA / "manifest.csv", encoding="utf-8") as f:
        return {row["track_id"]: row for row in csv.DictReader(f)}


def load_features(tid):
    """Same contents as FeatureExtractor.load_features: datasets plus attributes."""
    features = {}
    with h5py.File(DATA / "features" / f"{tid}.h5", "r") as f:
        for key in f.keys():
            features[key] = f[key][()]
        for key, value in f.attrs.items():
            features[key] = value
    return features


def stem_descriptions(tid, full_features):
    descs = []
    for stem in STEMS:
        path = DATA / "features" / "stems" / f"{tid}_{stem}.json"
        if not path.exists():
            continue
        with open(path) as fh:
            stem_feats = json.load(fh)
        descs.append(stem_comparison_description(full_features, stem_feats, stem))
    descs += ablation_description(tid)
    return "\n".join(descs)


ABLATION_MEASURES = (("rms_mean", "loudness"), ("onset_mean", "rhythmic drive (spectral flux)"),
                     ("centroid_weighted", "brightness (spectral centroid)"),
                     ("flatness_weighted", "noisiness (spectral flatness)"))


def ablation_description(tid):
    """What removing each stem does to the mix, from jobs/phase2/extract_ablation.pbs."""
    path = DATA / "features" / "ablation" / f"{tid}.json"
    if not path.exists():
        return []
    with open(path) as fh:
        abl = json.load(fh)
    full = abl["full"]
    lines = ["Measured effect of removing each stem from the mix:"]
    for stem in STEMS:
        minus = abl.get(f"minus_{stem}")
        if not minus:
            continue
        changes = []
        for key, label in ABLATION_MEASURES:
            if full.get(key):
                pct = (minus[key] - full[key]) / abs(full[key]) * 100
                changes.append(f"{label} {pct:+.0f}%")
        lines.append(f"  - without the {stem}: " + ", ".join(changes))
    return lines


def world_context(year, db_path):
    if year <= 0 or not db_path.exists():
        return ""
    ctx = get_world_context(year, "US", db_path)
    ctx["charts"] = ctx.get("charts", [])[:MAX_CHART_ROWS]
    return json.dumps(ctx, indent=2, default=str)


def main():
    manifest = load_manifest()
    all_ids = sorted(manifest)
    start = SHARD * TRACKS_PER_SHARD
    if start >= len(all_ids):
        print(f"Shard {SHARD} beyond manifest ({len(all_ids)} tracks).", flush=True)
        return 0
    slice_ids = all_ids[start:start + TRACKS_PER_SHARD]
    print(f"Annotating tracks {start}-{start + len(slice_ids)} ({len(slice_ids)} tracks)", flush=True)

    generator = ReasoningGenerator(ReasoningConfig(provider="anthropic", model_name="claude-haiku-5-5"))
    ann_dir = DATA / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    db_path = DATA / "world_context" / "world_context.db"

    done = failed = 0
    for tid in slice_ids:
        out = ann_dir / f"{tid}.json"
        if out.exists():
            done += 1
            continue
        if not (DATA / "features" / f"{tid}.h5").exists():
            print(f"  Skip {tid}: no features", flush=True)
            continue
        rec = manifest[tid]
        year = int(rec.get("year") or 0)
        features = load_features(tid)
        feat_desc = features_to_description(features)
        stem_desc = stem_descriptions(tid, features)
        try:
            world_ctx = world_context(year, db_path)
        except Exception as e:
            print(f"  world context failed for {tid} ({year}): {e}", flush=True)
            world_ctx = ""
        try:
            reasoning = generator.generate_from_features(
                f"Title: {rec['title']}, Artist: {rec['artist']}, Genre: {rec['genre']}, Year: {year}",
                feat_desc,
                stem_desc,
                world_ctx,
            )
        except Exception as e:
            print(f"  FAILED {tid}: {type(e).__name__}: {e}", flush=True)
            failed += 1
            continue
        stop_reason = getattr(generator, "last_stop_reason", None)
        if stop_reason != "end_turn":
            # Not saved, so a rerun retries this track instead of skipping a truncated file
            print(f"  FAILED {tid}: stop_reason={stop_reason} ({len(reasoning)} chars)", flush=True)
            failed += 1
            continue
        annotation = {
            "track_id": tid,
            "stop_reason": stop_reason,
            "reasoning": reasoning,
            "feature_description": feat_desc,
            "stem_descriptions": stem_desc,
            "world_context": world_ctx,
            "metadata": {"title": rec["title"], "artist": rec["artist"], "genre": rec["genre"], "year": year},
        }
        tmp = out.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(annotation, f, indent=2)
        tmp.replace(out)
        done += 1
        if done % 5 == 0:
            print(f"  {done}/{len(slice_ids)} done", flush=True)

    print(f"Shard {SHARD} complete: {done}/{len(slice_ids)} annotated, {failed} failed.", flush=True)
    missing = [t for t in all_ids if (DATA / "features" / f"{t}.h5").exists() and not (ann_dir / f"{t}.json").exists()]
    if missing:
        print(f"{len(missing)} tracks still missing annotations.", flush=True)
    else:
        (ann_dir / ".annotate_done").touch()
        print("All tracks with features are annotated. Wrote done marker.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
