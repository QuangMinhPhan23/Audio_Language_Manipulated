"""Phase 0 v2 — Steps 1-3: count stems, cut windows, render sparse mixes.

Usage:
  python -I scripts/phase0v2_stems.py count    # Step 1: stem inventory from MoisesDB zip
  python -I scripts/phase0v2_stems.py windows  # Step 2: find active windows per stem
  python -I scripts/phase0v2_stems.py render   # Step 3: render sparse mixes + measure tempo/mode

Env vars:
  DATA_ROOT   — data directory (default: data)
  MOISESDB_ZIP — path to MoisesDB zip on disk (required for count + windows)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

DATA = Path(os.environ.get("DATA_ROOT", "data"))
P0 = DATA / "phase0"
MOISES_TRACKS = DATA / "raw" / "moisesdb" / "tracks"
MIN_INSTRUMENT_SONGS = 25


def _load_jsonl(path):
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def _save_jsonl(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows)} rows -> {path}", flush=True)


def _open_zip():
    """Open MoisesDB zip — local path (MOISESDB_ZIP) or remote URL (MOISESDB_URL, ~83 GB, streamed via HTTP Range)."""
    import zipfile
    zpath = os.environ.get("MOISESDB_ZIP", "")
    if zpath and Path(zpath).exists():
        return zipfile.ZipFile(zpath)
    url = os.environ.get("MOISESDB_URL", "")
    if url:
        from musicwhy.data.remote_zip import open_remote_zip
        print(f"Opening remote zip: {url[:60]}...", flush=True)
        return open_remote_zip(url)
    sys.exit("Set MOISESDB_ZIP (local path) or MOISESDB_URL (remote link from music.ai/research)")


def step_count():
    """Step 1: count songs per melodic stem group in MoisesDB."""
    from musicwhy.stimuli.stems import (count_stems, pick_instruments,
                                        read_data_jsons_from_zip, stem_inventory)

    zf = _open_zip()
    tracks = read_data_jsons_from_zip(zf)
    print(f"Read {len(tracks)} tracks from zip", flush=True)

    inv = stem_inventory(tracks)
    _save_jsonl(inv, P0 / "stem_inventory.jsonl")

    counts = count_stems(tracks)
    kept = pick_instruments(counts, MIN_INSTRUMENT_SONGS)
    dropped = {k: v for k, v in counts.items() if k not in kept}

    print(f"\nKept instruments ({len(kept)}):", flush=True)
    for inst in kept:
        print(f"  {inst}: {counts[inst]} songs", flush=True)
    if dropped:
        print(f"Dropped (< {MIN_INSTRUMENT_SONGS} songs):", flush=True)
        for inst, n in sorted(dropped.items()):
            print(f"  {inst}: {n} songs", flush=True)

    summary = {"kept": {k: counts[k] for k in kept},
               "dropped": dropped,
               "min_songs": MIN_INSTRUMENT_SONGS}
    (P0 / "stem_counts.json").parent.mkdir(parents=True, exist_ok=True)
    (P0 / "stem_counts.json").write_text(json.dumps(summary, indent=2))
    ok = all(counts[k] >= MIN_INSTRUMENT_SONGS for k in kept)
    print(f"\nTest v2.1: all kept values >= {MIN_INSTRUMENT_SONGS} -- {'PASS' if ok else 'FAIL'}", flush=True)


def step_windows():
    """Step 2: find active 20s windows for each track+instrument combination."""
    from musicwhy.stimuli.stems import (find_windows, load_stem_audio,
                                        read_data_jsons_from_zip, classify_song_stems, SR)

    inv_path = P0 / "stem_inventory.jsonl"
    counts_path = P0 / "stem_counts.json"
    if not inv_path.exists() or not counts_path.exists():
        sys.exit("Run 'count' first")

    inv = _load_jsonl(inv_path)
    counts = json.loads(counts_path.read_text())
    kept_instruments = set(counts["kept"])

    zf = _open_zip()
    tracks_by_uuid = {}
    for name in sorted(zf.namelist()):
        import posixpath
        if posixpath.basename(name) == "data.json" and name.count("/") >= 1:
            uuid = posixpath.basename(posixpath.dirname(name))
            tracks_by_uuid[uuid] = json.loads(zf.read(name))

    windows = []
    for i, entry in enumerate(inv):
        uuid = entry["uuid"]
        data_json = tracks_by_uuid.get(uuid)
        if data_json is None:
            continue
        data_json["_uuid"] = uuid

        instruments = entry.get("instruments", {})
        for instrument, stem_names in instruments.items():
            if instrument not in kept_instruments:
                continue
            for stem_name in stem_names:
                lead_audio = load_stem_audio(zf, stem_name, data_json, SR)
                if lead_audio is None or len(lead_audio) < SR * 20:
                    continue

                vocal_audio = load_stem_audio(zf, "vocals", data_json, SR)

                wins = find_windows(lead_audio, SR, vocal_audio=vocal_audio)
                for wi, (start, end) in enumerate(wins):
                    windows.append({
                        "uuid": uuid,
                        "instrument": instrument,
                        "stem_name": stem_name,
                        "window_idx": wi,
                        "start_s": start / SR,
                        "end_s": end / SR,
                        "start_sample": start,
                        "end_sample": end,
                    })

        if (i + 1) % 20 == 0:
            print(f"  [{i+1}/{len(inv)}] {len(windows)} windows so far", flush=True)

    _save_jsonl(windows, P0 / "windows.jsonl")
    n_songs = len({w["uuid"] for w in windows})
    by_inst = {}
    for w in windows:
        by_inst[w["instrument"]] = by_inst.get(w["instrument"], 0) + 1
    print(f"\n{len(windows)} windows from {n_songs} songs", flush=True)
    for inst, n in sorted(by_inst.items()):
        print(f"  {inst}: {n}", flush=True)


def step_render():
    """Step 3: render sparse mixes (lead + drums + bass +/- vocals) and measure tempo/mode."""
    import numpy as np
    import soundfile as sf
    from musicwhy.stimuli.stems import (load_stem_audio, render_sparse_mix,
                                        lufs_normalise, save_clip, SR,
                                        read_data_jsons_from_zip)
    from musicwhy.stimuli.design import band_of

    windows_path = P0 / "windows.jsonl"
    if not windows_path.exists():
        sys.exit("Run 'windows' first")
    windows = _load_jsonl(windows_path)

    clips_dir = P0 / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    zf = _open_zip()
    tracks_by_uuid = {}
    for name in sorted(zf.namelist()):
        import posixpath
        if posixpath.basename(name) == "data.json" and name.count("/") >= 1:
            uuid = posixpath.basename(posixpath.dirname(name))
            tracks_by_uuid[uuid] = json.loads(zf.read(name))

    results = []
    for i, w in enumerate(windows):
        clip_id = f"{w['uuid']}_{w['instrument'].replace(' ', '_')}_{w['window_idx']}"
        out_vocal = clips_dir / f"{clip_id}_vocal.wav"
        out_inst = clips_dir / f"{clip_id}_instrumental.wav"

        if out_vocal.exists() and out_inst.exists():
            results.append({"clip_id": clip_id, "skipped": True, **w})
            continue

        uuid = w["uuid"]
        data_json = tracks_by_uuid.get(uuid)
        if data_json is None:
            continue
        data_json["_uuid"] = uuid

        start = w["start_sample"]
        end = w["end_sample"]
        window = (start, end)

        lead = load_stem_audio(zf, w["stem_name"], data_json, SR)
        drums, bass, vocals = None, None, None
        for stem in data_json.get("stems", []):
            sn = stem.get("stemName", "")
            if sn == "drums":
                drums = load_stem_audio(zf, sn, data_json, SR)
            elif sn == "bass":
                bass = load_stem_audio(zf, sn, data_json, SR)
            elif sn == "vocals":
                vocals = load_stem_audio(zf, sn, data_json, SR)

        stems = {"lead": lead, "drums": drums, "bass": bass, "vocals": vocals}

        for include_vocals, suffix, path in [(True, "vocal", out_vocal),
                                             (False, "instrumental", out_inst)]:
            mix = render_sparse_mix(stems, window, include_vocals)
            mix, gain = lufs_normalise(mix, SR)
            save_clip(mix, SR, path)

        results.append({
            "clip_id": clip_id,
            "uuid": w["uuid"],
            "instrument": w["instrument"],
            "stem_name": w["stem_name"],
            "window_idx": w["window_idx"],
            "start_s": w["start_s"],
            "end_s": w["end_s"],
            "vocal_path": str(out_vocal),
            "instrumental_path": str(out_inst),
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(windows)}] rendered", flush=True)

    _save_jsonl(results, P0 / "rendered_clips.jsonl")
    print(f"\nRendered {len(results)} clip pairs", flush=True)


def _librosa_tempo(audio, sr):
    """Estimate BPM with librosa's beat tracker. Returns float or None."""
    import librosa
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr, units="time")
    return float(tempo[0]) if hasattr(tempo, '__len__') else (float(tempo) if tempo > 0 else None)


def _librosa_tempo_onset(audio, sr):
    """Second tempo estimate using onset-based beat tracking for cross-validation."""
    import librosa
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
    return float(tempo[0]) if len(tempo) > 0 and tempo[0] > 0 else None


KS_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KS_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def _librosa_mode(audio, sr):
    """Estimate major/minor using Krumhansl-Schmuckler key profiles on librosa chroma."""
    import librosa
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    chroma_sum = chroma.mean(axis=1)
    best_major = max(np.corrcoef(np.roll(KS_MAJOR, i), chroma_sum)[0, 1] for i in range(12))
    best_minor = max(np.corrcoef(np.roll(KS_MINOR, i), chroma_sum)[0, 1] for i in range(12))
    return "major" if best_major >= best_minor else "minor"


def _librosa_mode_stft(audio, sr):
    """Second mode estimate using STFT-based chroma for cross-validation."""
    import librosa
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    chroma_sum = chroma.mean(axis=1)
    best_major = max(np.corrcoef(np.roll(KS_MAJOR, i), chroma_sum)[0, 1] for i in range(12))
    best_minor = max(np.corrcoef(np.roll(KS_MINOR, i), chroma_sum)[0, 1] for i in range(12))
    return "major" if best_major >= best_minor else "minor"


def step_measure():
    """Measure tempo and mode for rendered clips using librosa (no essentia needed)."""
    import librosa
    import soundfile as sf
    from musicwhy.stimuli.design import band_of

    rendered_path = P0 / "rendered_clips.jsonl"
    if not rendered_path.exists():
        sys.exit("Run 'render' first")
    clips = _load_jsonl(rendered_path)

    clips_dir = P0 / "clips"
    results = []
    for i, clip in enumerate(clips):
        clip_id = clip["clip_id"]
        vocal_path = Path(clip.get("vocal_path", str(clips_dir / f"{clip_id}_vocal.wav")))
        if not vocal_path.exists():
            continue

        audio, file_sr = sf.read(str(vocal_path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if file_sr != 22050:
            audio_22k = librosa.resample(audio, orig_sr=file_sr, target_sr=22050, res_type="soxr_hq")
        else:
            audio_22k = audio
        audio_44k = audio if file_sr == 44100 else librosa.resample(audio, orig_sr=file_sr, target_sr=44100, res_type="soxr_hq")

        t_a = _librosa_tempo(audio_22k, 22050)
        t_b = _librosa_tempo_onset(audio_22k, 22050)
        mode_a = _librosa_mode(audio_44k, 44100)
        mode_b = _librosa_mode_stft(audio_44k, 44100)

        tempo_agree = (t_a is not None and t_b is not None and
                       band_of(t_a) is not None and band_of(t_a) == band_of(t_b))
        mode_agree = mode_a == mode_b

        results.append({
            **clip,
            "tempo": {"librosa_beat": t_a, "librosa_onset": t_b},
            "tempo_band": band_of(t_a) if tempo_agree else None,
            "mode": mode_a if mode_agree else None,
            "tempo_agree": tempo_agree,
            "mode_agree": mode_agree,
        })
        if (i + 1) % 10 == 0:
            print(f"  [{i+1}/{len(clips)}] measured", flush=True)

    _save_jsonl(results, P0 / "measured_clips.jsonl")
    n_tempo = sum(1 for r in results if r["tempo_band"] is not None)
    n_mode = sum(1 for r in results if r["mode"] is not None)
    print(f"\n{len(results)} clips measured: tempo agreed on {n_tempo}, mode agreed on {n_mode}", flush=True)


if __name__ == "__main__":
    cmds = {"count": step_count, "windows": step_windows, "render": step_render, "measure": step_measure}
    if len(sys.argv) < 2 or sys.argv[1] not in cmds:
        print(__doc__)
        print(f"Commands: {', '.join(cmds)}")
        sys.exit(1)
    cmds[sys.argv[1]]()
