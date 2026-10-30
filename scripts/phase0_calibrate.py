"""Phase 0 Test 4.1: check the measuring tools on clips whose answers are known before trusting them.

Two calibration sets, built in the verify environment:
  midi   50 clips rendered from generated MIDI with a General MIDI SoundFont: known tempo, mode and lead
         instrument (one of the six), plus bass and drums. Rendered audio is easier than real recordings.
  musdb  25 + 25 twenty-second excerpts of MUSDB18-HQ test songs: the mixture (vocals present) and the sum of
         drums, bass and other over the same seconds (no vocals), so the vocals answer is exact.
Writes $DATA_ROOT/phase0/calibration/{clips,results.json}. Pass: both tempo tools and both key profiles get the
tempo band and the mode right on at least 85% of the MIDI clips.

  python scripts/phase0_calibrate.py [--soundfont PATH]
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from musicwhy.stimuli import design  # noqa: E402
from musicwhy.stimuli.acceptance import THRESHOLDS, check_attributes  # noqa: E402
from musicwhy.stimuli.measure import MODEL_DIR, Measurer  # noqa: E402

DATA = Path(os.environ.get("DATA_ROOT", "data"))
CAL = DATA / "phase0" / "calibration"
SR = 44100
GM_PROGRAM = {"piano": 0, "electric guitar": 27, "saxophone": 65, "synthesizer": 81, "violin": 40, "trumpet": 56}
MAJOR, MINOR = [0, 2, 4, 5, 7, 9, 11], [0, 2, 3, 5, 7, 8, 10]
TONIC_MIDI = {t: 60 + i for i, t in enumerate(design.TONICS)}


def midi_events(spec, seconds=20.0):
    """(time, chan, key, velocity) note-on/off events: lead melody (chan 0), bass (1), drums (9).
    Chords I-IV-V-I in major or i-iv-v-i in natural minor, one per bar."""
    rng = random.Random(spec["seed"])
    scale = MAJOR if spec["mode"] == "major" else MINOR
    tonic = TONIC_MIDI[spec["tonic"]]
    beat = 60.0 / spec["bpm"]
    ev = []

    def note(t, chan, key, dur, vel):
        if t < seconds:
            ev.append((t, chan, key, vel))
            ev.append((min(t + dur, seconds), chan, key, 0))

    degree_of_bar = [0, 3, 4, 0]  # I IV V I
    n_beats = int(seconds / beat)
    for b in range(n_beats):
        t, bar_deg = b * beat, degree_of_bar[(b // 4) % 4]
        triad = [tonic + scale[(bar_deg + k) % 7] + 12 * ((bar_deg + k) // 7) for k in (0, 2, 4)]
        note(t, 1, triad[0] - 24, beat * 0.9, 90)  # bass on the chord root
        note(t, 9, 36 if b % 2 == 0 else 38, beat * 0.5, 100)  # kick on 1 and 3, snare on 2 and 4
        for half in (0, 0.5):
            note(t + half * beat, 9, 42, beat * 0.25, 70)  # closed hi-hat on eighths
        # Lead: a chord tone on the beat, a scale step on the off-beat
        note(t, 0, rng.choice(triad) + 12 * (spec["instrument"] in ("violin", "trumpet")), beat * 0.48, 105)
        step = tonic + scale[rng.randrange(7)] + 12 * (spec["instrument"] in ("violin", "trumpet"))
        note(t + beat / 2, 0, step, beat * 0.45, 95)
    return sorted(ev)


def render(spec, soundfont, path, seconds=20.0):
    import soundfile as sf
    import tinysoundfont

    synth = tinysoundfont.Synth(samplerate=SR)
    sfid = synth.sfload(str(soundfont))
    synth.program_select(0, sfid, 0, GM_PROGRAM[spec["instrument"]])
    synth.program_select(1, sfid, 0, 33)  # electric bass (finger)
    synth.program_select(9, sfid, 0, 0, True)
    chunks, now = [], 0.0
    for t, chan, key, vel in midi_events(spec, seconds) + [(seconds, None, None, None)]:
        n = int(round((t - now) * SR))
        if n > 0:
            chunks.append(np.frombuffer(synth.generate(n), dtype=np.float32).reshape(-1, 2).copy())
            now += n / SR
        if chan is not None:
            synth.noteon(chan, key, vel) if vel else synth.noteoff(chan, key)
    audio = np.concatenate(chunks)
    sf.write(path, audio / max(np.abs(audio).max(), 1e-9) * 0.9, SR)


def midi_set(soundfont, n=50):
    specs = []
    for i in range(n):
        rng = random.Random(design.stable_seed("calibration", i))
        band = list(design.TEMPO_BANDS)[i % 3]
        lo, hi = design.TEMPO_BANDS[band]
        specs.append({"clip_id": f"midi_{i:02d}", "seed": design.stable_seed("calibration", i),
                      "instrument": design.INSTRUMENTS[i % 6], "mode": design.MODES[(i // 6) % 2],
                      "tonic": rng.choice(design.TONICS), "bpm": rng.randint(lo, hi), "band": band})
    for s in specs:
        s["path"] = str(CAL / "clips" / f"{s['clip_id']}.wav")
        if not Path(s["path"]).exists():
            render(s, soundfont, s["path"])
    return specs


def musdb_set(n=25, seconds=20.0):
    import soundfile as sf

    tracks = sorted(p.parent for p in (DATA / "raw" / "musdb18hq").glob("test/*/vocals.wav"))
    tracks = random.Random(0).sample(tracks, min(n, len(tracks)))
    specs = []
    for tr in tracks:
        name = tr.name.replace(" ", "_")
        vox, sr = sf.read(tr / "vocals.wav", dtype="float32")
        win, hop = int(seconds * sr), sr
        energy = [float(np.mean(vox[s:s + win] ** 2)) for s in range(0, len(vox) - win, hop)]
        start = int(np.argmax(energy)) * hop  # the 20 seconds with the most singing
        mix = sf.read(tr / "mixture.wav", dtype="float32", start=start, stop=start + win)[0]
        inst = sum(sf.read(tr / f"{s}.wav", dtype="float32", start=start, stop=start + win)[0]
                   for s in ("drums", "bass", "other"))
        for label, audio in (("vocal", mix), ("instrumental", inst)):
            path = CAL / "clips" / f"musdb_{label}_{name}.wav"
            if not path.exists():
                sf.write(path, audio, sr)
            specs.append({"clip_id": path.stem, "vocals": label, "path": str(path)})
    return specs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--soundfont", default=str(MODEL_DIR / "MuseScore_General.sf2"))
    args = p.parse_args()
    (CAL / "clips").mkdir(parents=True, exist_ok=True)
    m = Measurer()
    rows = []
    for s in midi_set(args.soundfont):
        meas = m.measure_clip(s["path"])
        target = {"tempo_bpm": s["bpm"], "mode": s["mode"], "lead_instrument": s["instrument"],
                  "vocals": "instrumental"}
        rows.append({"set": "midi", **s, "measurements": meas, "checks": check_attributes(target, meas)})
    for s in musdb_set():
        rows.append({"set": "musdb", **s, "vocal_ratio": m.vocal_ratio(s["path"])})

    midi = [r for r in rows if r["set"] == "midi"]
    summary = {"tempo_tools": [m.tempo_a, m.tempo_b], "n_midi": len(midi)}
    for tool in midi[0]["measurements"]["tempo"]:
        summary[f"tempo_band_acc_{tool}"] = float(np.mean(
            [design.band_of(r["measurements"]["tempo"][tool]) == r["band"] for r in midi]))
    for prof in midi[0]["measurements"]["mode"]:
        summary[f"mode_acc_{prof}"] = float(np.mean([r["measurements"]["mode"][prof] == r["mode"] for r in midi]))
    for tagger in midi[0]["measurements"]["instrument"]:
        probs = [r["measurements"]["instrument"][tagger] for r in midi]
        summary[f"instrument_top1_{tagger}"] = float(np.mean(
            [max(p, key=p.get) == r["instrument"] for p, r in zip(probs, midi)]))
    for attr in design.ATTRIBUTES:
        summary[f"rule_pass_{attr}"] = float(np.mean([r["checks"][attr]["pass"] for r in midi]))
    mus = [r for r in rows if r["set"] == "musdb"]
    for label in ("vocal", "instrumental"):
        ratios = sorted(r["vocal_ratio"] for r in mus if r["vocals"] == label)
        summary[f"vocal_ratio_{label}"] = {"min": ratios[0], "median": ratios[len(ratios) // 2], "max": ratios[-1]}
    summary["vocals_acc_at_thresholds"] = float(np.mean(
        [(r["vocal_ratio"] < THRESHOLDS["vocal_max_instrumental"]) if r["vocals"] == "instrumental"
         else (r["vocal_ratio"] > THRESHOLDS["vocal_min_vocal"]) for r in mus]))
    keys = [k for k in summary if k.startswith(("tempo_band_acc", "mode_acc"))]
    summary["test_4_1_pass"] = all(summary[k] >= 0.85 for k in keys)
    (CAL / "results.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
