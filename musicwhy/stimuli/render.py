"""Render MIDI clips with a General MIDI SoundFont — instrument, tempo and mode are exact by construction.

Extracted from scripts/phase0_calibrate.py so the generation pipeline can use it for instrument, tempo and
mode clips (ACE-Step only needed for vocals clips). Runs in the verify environment (tinysoundfont).
"""
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from musicwhy.stimuli import design

SR = 44100
GM_PROGRAM: Dict[str, int] = {
    "piano": 0, "electric guitar": 27, "saxophone": 65,
    "synthesizer": 81, "violin": 40, "trumpet": 56,
}
MAJOR = [0, 2, 4, 5, 7, 9, 11]
MINOR = [0, 2, 3, 5, 7, 8, 10]
TONIC_MIDI = {t: 60 + i for i, t in enumerate(design.TONICS)}


def midi_events(spec: dict, seconds: float = 20.0) -> List[Tuple[float, int, int, int]]:
    """(time, channel, key, velocity) note-on/off events: lead melody (ch 0), bass (ch 1), drums (ch 9).
    Chords I–IV–V–I in major or i–iv–v–i in natural minor, one chord per bar."""
    rng = random.Random(spec["seed"])
    scale = MAJOR if spec["mode"] == "major" else MINOR
    tonic = TONIC_MIDI[spec["tonic"]]
    beat = 60.0 / spec["bpm"]
    ev: list = []

    def note(t, chan, key, dur, vel):
        if t < seconds:
            ev.append((t, chan, key, vel))
            ev.append((min(t + dur, seconds), chan, key, 0))

    degree_of_bar = [0, 3, 4, 0]  # I IV V I
    n_beats = int(seconds / beat)
    for b in range(n_beats):
        t, bar_deg = b * beat, degree_of_bar[(b // 4) % 4]
        triad = [tonic + scale[(bar_deg + k) % 7] + 12 * ((bar_deg + k) // 7) for k in (0, 2, 4)]
        note(t, 1, triad[0] - 24, beat * 0.9, 90)  # bass
        note(t, 9, 36 if b % 2 == 0 else 38, beat * 0.5, 100)  # kick/snare
        for half in (0, 0.5):
            note(t + half * beat, 9, 42, beat * 0.25, 70)  # hi-hat
        # Lead: chord tone on beat, scale step on off-beat; violin/trumpet up an octave
        octave = 12 if spec["instrument"] in ("violin", "trumpet") else 0
        note(t, 0, rng.choice(triad) + octave, beat * 0.48, 105)
        note(t + beat / 2, 0, tonic + scale[rng.randrange(7)] + octave, beat * 0.45, 95)
    return sorted(ev)


def render_midi(spec: dict, soundfont: str, path, seconds: float = 20.0) -> None:
    """Render a MIDI clip to a WAV file. *spec* must have keys: seed, instrument, mode, tonic, bpm."""
    import soundfile as sf
    import tinysoundfont

    path = Path(path)
    synth = tinysoundfont.Synth(samplerate=SR)
    sfid = synth.sfload(str(soundfont))
    synth.program_select(0, sfid, 0, GM_PROGRAM[spec["instrument"]])
    synth.program_select(1, sfid, 0, 33)  # electric bass (finger)
    synth.program_select(9, sfid, 0, 0, True)  # drums
    chunks, now = [], 0.0
    for t, chan, key, vel in midi_events(spec, seconds) + [(seconds, None, None, None)]:
        n = int(round((t - now) * SR))
        if n > 0:
            chunks.append(np.frombuffer(synth.generate(n), dtype=np.float32).reshape(-1, 2).copy())
            now += n / SR
        if chan is not None:
            synth.noteon(chan, key, vel) if vel else synth.noteoff(chan, key)
    audio = np.concatenate(chunks)
    sf.write(str(path), audio / max(np.abs(audio).max(), 1e-9) * 0.9, SR)


def render_params(clip: dict) -> dict:
    """Build the spec dict that render_midi expects from a clip_plan entry."""
    t = clip["target"]
    return {
        "seed": clip["seed"],
        "instrument": t["lead_instrument"],
        "mode": t["mode"],
        "tonic": t["keyscale"].split()[0],
        "bpm": t["tempo_bpm"],
    }
