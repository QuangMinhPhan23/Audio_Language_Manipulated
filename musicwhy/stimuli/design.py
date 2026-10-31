"""Attribute design for fictional songs (PHASE0_stimuli_pipeline.md §1).

Stdlib only. Every random choice is seeded from the song id, so rebuilding gives the same songs, conflict
values and generation seeds.

v1 instruments (ACE-Step/MIDI): piano, electric guitar, saxophone, synthesizer, violin, trumpet
v2 instruments (real MoisesDB stems): piano, electric guitar, acoustic guitar, synthesizer, strings, organ
   — final set fixed after step 1 stem counts; any value with < 25 songs is dropped.
"""
import itertools
import random
import zlib
from typing import Any, Dict, List, Optional

INSTRUMENTS = ["piano", "electric guitar", "saxophone", "synthesizer", "violin", "trumpet"]
INSTRUMENTS_V2 = ["piano", "electric guitar", "acoustic guitar", "synthesizer", "strings", "organ", "wind"]
# Inside ACE-Step's stable range; its docs warn that extreme tempos are unstable
TEMPO_BANDS = {"slow": (70, 85), "medium": (100, 115), "fast": (130, 145)}
MODES = ["major", "minor"]
VOCALS = ["instrumental", "vocal"]
ATTRIBUTES = ("lead_instrument", "tempo_band", "mode", "vocals")
VALUES = {"lead_instrument": INSTRUMENTS, "tempo_band": list(TEMPO_BANDS), "mode": MODES, "vocals": VOCALS}
VALUES_V2 = {"lead_instrument": INSTRUMENTS_V2, "tempo_band": list(TEMPO_BANDS), "mode": MODES, "vocals": VOCALS}
# ACE-Step's VALID_KEYSCALES are "<note><accidental> <major|minor>" (acestep/constants.py)
TONICS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Original, deliberately plain lines for the vocal clips: no real lyrics, titles or names
NEUTRAL_LYRICS = [
    "[Verse]\nMorning light across the floor\nSlow steps toward the open door\n"
    "[Chorus]\nHold on, the day is new\nHold on, I'm coming through",
    "[Verse]\nRain along the station wall\nQuiet voices down the hall\n"
    "[Chorus]\nWe keep on walking home\nWe keep on walking home",
    "[Verse]\nPaper boats on a winter stream\nHalf awake inside a dream\n"
    "[Chorus]\nCarry me over the line\nCarry me, one more time",
    "[Verse]\nStreetlights humming one by one\nCounting hours till the night is done\n"
    "[Chorus]\nSay it slow, say it clear\nEverything I need is here",
]


def stable_seed(*parts: Any) -> int:
    """A non-negative 31-bit seed from the parts' text, the same on every machine and Python version."""
    return zlib.crc32("|".join(map(str, parts)).encode()) & 0x7FFFFFFF


def assign_attributes(n: int = 500, seed: int = 0) -> List[Dict[str, Any]]:
    """n songs, each one cell of the 6×3×2×2 design: shuffled full copies of the 72 cells, cut to n.

    Cutting the last copy short drops at most 72 − (n mod 72) cells, so every value stays within a few songs
    of n / (number of values) (Test 1.1 allows ±5).
    """
    rng = random.Random(seed)
    cells = list(itertools.product(*(VALUES[a] for a in ATTRIBUTES)))
    rows: List[tuple] = []
    while len(rows) < n:
        block = cells[:]
        rng.shuffle(block)
        rows.extend(block)
    songs = []
    for i, cell in enumerate(rows[:n]):
        song = {"song_id": f"fs{i:04d}", **dict(zip(ATTRIBUTES, cell))}
        lo, hi = TEMPO_BANDS[song["tempo_band"]]
        song["tempo_bpm"] = random.Random(stable_seed(song["song_id"], "bpm")).randint(lo, hi)
        songs.append(song)
    return songs


def conflict_song(song: Dict[str, Any], attr: str) -> Dict[str, Any]:
    """The song with one attribute changed to a different value, fixed per (song, attribute) across seeds."""
    rng = random.Random(stable_seed(song["song_id"], "conflict", attr))
    out = dict(song)
    out[attr] = rng.choice([v for v in VALUES[attr] if v != song[attr]])
    if attr == "tempo_band":
        lo, hi = TEMPO_BANDS[out["tempo_band"]]
        out["tempo_bpm"] = rng.randint(lo, hi)
    return out


def keyscale(mode: str, seed: int) -> str:
    """ACE-Step key string; the tonic varies with the seed because only the mode is a song attribute."""
    return f"{TONICS[seed % 12]} {mode}"


def caption(target: Dict[str, Any]) -> str:
    """Caption with instrument and vocals only. ACE-Step's Tutorial says tempo and key belong in the bpm and
    keyscale parameters, not in the caption, so the mode is set through keyscale alone."""
    voice = "instrumental" if target["vocals"] == "instrumental" else "with vocals"
    return f"{target['lead_instrument']} lead, {voice}, steady beat"


def clip_plan(song: Dict[str, Any], n_per_arm: int = 2) -> List[Dict[str, Any]]:
    """Per-attribute clips for one song: n_per_arm agree + n_per_arm conflict per attribute.

    Instrument, tempo and mode clips are always MIDI (exact by construction, no vocals needed).
    Vocals clips use ACE-Step when the target is "vocal", MIDI when "instrumental".
    """
    plan = []
    for attr in ATTRIBUTES:
        for arm in ("agree", "conflict"):
            for k in range(n_per_arm):
                base = conflict_song(song, attr) if arm == "conflict" else dict(song)
                seed = stable_seed(song["song_id"], attr, arm, k)
                target = {a: base[a] for a in (*ATTRIBUTES, "tempo_bpm")}
                target["keyscale"] = keyscale(target["mode"], seed)
                if attr == "vocals":
                    source = "acestep" if target["vocals"] == "vocal" else "midi"
                else:
                    source = "midi"
                plan.append({
                    "clip_id": f"{song['song_id']}_{attr}_{arm}_{k}",
                    "song_id": song["song_id"],
                    "test_attr": attr,
                    "arm": arm,
                    "conflict_attr": attr if arm == "conflict" else None,
                    "seed": seed,
                    "target": target,
                    "source": source,
                })
    return plan


def generation_params(clip: Dict[str, Any], duration: float = 20.0) -> Dict[str, Any]:
    """Keyword arguments for ACE-Step's GenerationParams. The planner stays off: thinking and every CoT
    option are False, and phase0_generate.py also passes llm_handler=None."""
    t = clip["target"]
    instrumental = t["vocals"] == "instrumental"
    lyrics = "[Instrumental]" if instrumental else NEUTRAL_LYRICS[clip["seed"] % len(NEUTRAL_LYRICS)]
    return {
        "task_type": "text2music",
        "caption": caption(t),
        "lyrics": lyrics,
        "instrumental": instrumental,
        "vocal_language": "unknown" if instrumental else "en",
        "bpm": int(t["tempo_bpm"]),
        "keyscale": t["keyscale"],
        "timesignature": "4",
        "duration": duration,
        "inference_steps": 8,
        "shift": 3.0,
        "infer_method": "ode",
        "seed": clip["seed"],
        "thinking": False,
        "use_cot_metas": False,
        "use_cot_caption": False,
        "use_cot_language": False,
    }


def value_counts(songs: List[Dict[str, Any]], attr: str) -> Dict[str, int]:
    counts = {v: 0 for v in VALUES[attr]}
    for s in songs:
        counts[s[attr]] += 1
    return counts


def band_of(bpm: Optional[float]) -> Optional[str]:
    """The tempo band a measured tempo falls in, or None if it falls between bands."""
    if bpm is None:
        return None
    for band, (lo, hi) in TEMPO_BANDS.items():
        if lo <= bpm <= hi:
            return band
    return None
