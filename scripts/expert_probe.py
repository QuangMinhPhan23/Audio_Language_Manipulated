"""Experiment 2 on MGPHot clips: which representation predicts the experts' ratings better?

Compares (a) our 7 raw audio measurements with (b) MERT embeddings of the same 30-second clips, using
ridge regression and 5-fold cross-validation over songs. Reports R² per expert rating (0 = no better
than predicting the average, 1 = perfect). Also (c) the year alone, to show how much is just era.

  cd /srv/scratch/z5531827/musicwhy && .venv/bin/python -I scripts/expert_probe.py
Inputs: data/raw/mgphot/songs_with_clips.json, data/embeddings/mgphot_mert.npz.
Output: data/results/expert_probe.json
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path(__file__).resolve().parents[1] / "data"
RAW = ["tempo_bpm", "major_mode", "harmonic_percussive_ratio", "onset_strength", "rms_mean",
       "centroid_weighted", "dynamic_range_db"]
GENES = ["Synthetic Sonority", "Acoustic Sonority", "Electric Sonority", "Synthetic Drums", "Drum Aggressiveness",
         "Electric Guitar Distortion", "Electric Guitar", "Acoustic Guitar", "Piano", "Synthesizer",
         "String Ensemble", "Horn Ensemble", "Reed Instrument", "Minor / Major Key Tonality", "Tempo",
         "Danceability", "Aural Intensity", "Swing Feel", "Vocal Grittiness", "Vocal Breathiness"]


def r2_cv(model, X, y, kf):
    pred = cross_val_predict(model, X, y, cv=kf)
    return float(1 - np.mean((y - pred) ** 2) / np.var(y))


def main():
    songs = json.loads((DATA / "raw" / "mgphot" / "songs_with_clips.json").read_text())
    emb = np.load(DATA / "embeddings" / "mgphot_mert.npz")
    songs = [s for s in songs if s["youtube_id"] in emb.files]
    print(f"{len(songs)} songs with both measurements and MERT embeddings", flush=True)
    raw = np.array([[0.5 if s["measured"][f] is None else s["measured"][f] for f in RAW] for s in songs], float)
    mert = np.stack([np.asarray(emb[s["youtube_id"]], float).ravel() for s in songs])
    decade = np.array([[s["year"] // 10 * 10] for s in songs])
    kf = KFold(5, shuffle=True, random_state=0)
    alphas = np.logspace(-2, 4, 13)
    models = {
        "decade only": (make_pipeline(OneHotEncoder(handle_unknown="ignore"), RidgeCV(alphas=alphas)), decade),
        "raw measurements": (make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)), raw),
        "MERT embedding": (make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)), mert),
        "MERT + decade": (make_pipeline(StandardScaler(), RidgeCV(alphas=alphas)),
                          np.hstack([mert, OneHotEncoder(sparse_output=False).fit_transform(decade)])),
    }
    results = {}
    print(f"{'expert rating':28}" + "".join(f"{m:>18}" for m in models))
    for g in GENES:
        y = np.array([s["genes"][g] for s in songs])
        results[g] = {m: r2_cv(model, X, y, kf) for m, (model, X) in models.items()}
        print(f"{g:28}" + "".join(f"{results[g][m]:18.2f}" for m in models), flush=True)
    means = {m: float(np.mean([results[g][m] for g in GENES])) for m in models}
    print(f"{'mean':28}" + "".join(f"{means[m]:18.2f}" for m in models))
    out = DATA / "results" / "expert_probe.json"
    out.write_text(json.dumps({"n_songs": len(songs), "r2": results, "mean_r2": means}, indent=1))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
