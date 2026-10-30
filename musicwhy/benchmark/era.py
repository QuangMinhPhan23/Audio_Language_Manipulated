"""Task 5 (sound-world connection) built from MGPHot expert attributes of Billboard Hot 100 hits.

Every multiple-choice answer is measured, not written: decade means of expert ratings over all
21,299 songs, or a song's own decade. The open-ended "why" questions are graded against
checklists of sourced explanations (data/benchmark/era_checklists.json), kept separate from the
measured "what changed" part so a wrong explanation cannot hide behind a right trend.
Song titles and artists never appear in questions, so answers can't come from remembering a hit.
"""

from __future__ import annotations

import math
import random
from typing import Any, Dict, List, Sequence, Tuple

from musicwhy.data.mgphot import by_decade

# Sound attributes used in "what changed" / "peak decade" questions, in plain words of similar length
# (so the correct option is not given away by being longer).
SOUND_TERMS: Dict[str, str] = {
    "Synthetic Sonority": "electronic production",
    "Synthetic Drums": "drum machines",
    "Synthesizer": "synthesizers",
    "Acoustic Sonority": "an acoustic sound",
    "Horn Ensemble": "horn sections",
    "String Ensemble": "string sections",
    "Electric Guitar": "electric guitar",
    "Electric Guitar Distortion": "distorted electric guitar",
    "Piano": "piano",
    "Organ": "organ",
    "Rhodes": "Rhodes electric piano",
    "Reed Instrument": "saxophone",
    "Swing Feel": "a swing rhythm",
    "Compound Meter": "6/8 time",
    "Percussion": "hand percussion",
    "Acoustic Guitar": "acoustic guitar",
    "Backbeat": "a strong backbeat",
    "Focus on Riffs": "songs built on riffs",
}
DECADE_NAMES = {1950: "1950s", 1960: "1960s", 1970: "1970s", 1980: "1980s", 1990: "1990s",
                2000: "2000s", 2010: "2010s", 2020: "2020s"}
LETTERS = "ABCD"

# A change counts when it is at least this large on the 0-1 scale and many standard errors from zero
MIN_CHANGE = 0.10
MIN_Z = 4.0
# Distractors must have changed by less than this in the asked direction
MAX_DISTRACTOR_CHANGE = 0.02
# Peak-decade questions: every wrong decade must sit at least this far below the peak
PEAK_MARGIN = 0.08
# At most this many "what changed" questions share the same attribute and direction
MAX_PER_TOPIC = 3


def _mean_var(values: Sequence[float]) -> Tuple[float, float]:
    n = len(values)
    m = sum(values) / n
    v = sum((x - m) ** 2 for x in values) / max(n - 1, 1)
    return m, v


def decade_stats(songs: List[Dict[str, Any]], genes: Sequence[str]) -> Dict[int, Dict[str, Tuple[float, float, int]]]:
    """decade -> gene -> (mean, variance, n)."""
    out: Dict[int, Dict[str, Tuple[float, float, int]]] = {}
    for d, rs in by_decade(songs).items():
        out[d] = {}
        for g in genes:
            m, v = _mean_var([r["genes"][g] for r in rs])
            out[d][g] = (m, v, len(rs))
    return out


def _change(stats, d1: int, d2: int, gene: str) -> Tuple[float, float]:
    """(difference of means d2 - d1, z-score)."""
    m1, v1, n1 = stats[d1][gene]
    m2, v2, n2 = stats[d2][gene]
    se = math.sqrt(v1 / n1 + v2 / n2) or 1e-9
    return m2 - m1, (m2 - m1) / se


def _pick_distractors(rng: random.Random, correct: str, pool: List[str], k: int = 3) -> List[str]:
    """k random distractors, swapping one for a longer term when the answer would be the longest option."""
    wrong = rng.sample(pool, k)
    if all(len(w) < len(correct) for w in wrong):
        longer = [p for p in pool if len(p) >= len(correct) and p not in wrong]
        if longer:
            wrong[rng.randrange(k)] = rng.choice(longer)
    return wrong


def _shuffle_options(rng: random.Random, correct: str, wrong: List[str]) -> Tuple[Dict[str, str], str]:
    opts = [correct] + wrong
    rng.shuffle(opts)
    return dict(zip(LETTERS, opts)), LETTERS[opts.index(correct)]


def change_questions(songs: List[Dict[str, Any]], seed: int = 0) -> List[Dict[str, Any]]:
    """'Which became clearly more (less) common between decade X and Y?' with exactly one option that did."""
    rng = random.Random(seed)
    stats = decade_stats(songs, list(SOUND_TERMS))
    decades = sorted(stats)
    questions = []
    per_topic: Dict[Tuple[str, int], int] = {}
    term_gene = {t: g for g, t in SOUND_TERMS.items()}
    for i, d1 in enumerate(decades):
        for d2 in decades[i + 1:i + 4]:  # gaps of 10, 20 and 30 years
            for direction in (1, -1):
                moves = {g: _change(stats, d1, d2, g) for g in SOUND_TERMS}
                movers = [g for g, (dm, z) in moves.items()
                          if direction * dm >= MIN_CHANGE and direction * z >= MIN_Z
                          and per_topic.get((g, direction), 0) < MAX_PER_TOPIC]
                still = [g for g, (dm, _) in moves.items() if direction * dm < MAX_DISTRACTOR_CHANGE]
                if not movers or len(still) < 3:
                    continue
                correct = max(movers, key=lambda g: direction * moves[g][0])
                per_topic[(correct, direction)] = per_topic.get((correct, direction), 0) + 1
                wrong_terms = _pick_distractors(rng, SOUND_TERMS[correct], [SOUND_TERMS[g] for g in still])
                wrong = [term_gene[t] for t in wrong_terms]
                options, answer = _shuffle_options(rng, SOUND_TERMS[correct], wrong_terms)
                word = "more" if direction > 0 else "less"
                m1, m2 = stats[d1][correct][0], stats[d2][correct][0]
                explain = [f"{SOUND_TERMS[correct]}: {m1:.2f} -> {m2:.2f}"] + [
                    f"{SOUND_TERMS[g]}: {stats[d1][g][0]:.2f} -> {stats[d2][g][0]:.2f}" for g in wrong]
                questions.append({
                    "task_id": 5,
                    "question_type": "era_change",
                    "modality": "text",
                    "question_text": (
                        f"Experts rated the sound of every Billboard Hot 100 hit. Between the "
                        f"{DECADE_NAMES[d1]} and the {DECADE_NAMES[d2]}, which ONE of these became clearly "
                        f"{word} common in hits?"
                    ),
                    "audio_reference": "",
                    "options": options,
                    "correct_answer": answer,
                    "reference_explanation": (
                        f"Average expert rating (0-1) across Hot 100 hits, {DECADE_NAMES[d1]} -> "
                        f"{DECADE_NAMES[d2]}: " + "; ".join(explain) + "."
                    ),
                    "metadata": {"decade_from": d1, "decade_to": d2, "direction": word, "gene": correct,
                                 "change": round(moves[correct][0], 3), "z": round(moves[correct][1], 1),
                                 "n_from": stats[d1][correct][2], "n_to": stats[d2][correct][2]},
                })
    return questions


def peak_questions(songs: List[Dict[str, Any]], seed: int = 0) -> List[Dict[str, Any]]:
    """'In which decade were X most common?' where all three wrong decades sit well below the peak."""
    rng = random.Random(seed + 1)
    stats = decade_stats(songs, list(SOUND_TERMS))
    decades = sorted(stats)
    questions = []
    for g, term in SOUND_TERMS.items():
        means = {d: stats[d][g][0] for d in decades}
        peak = max(means, key=means.get)
        low = [d for d in decades if means[d] <= means[peak] - PEAK_MARGIN]
        if len(low) < 3:
            continue
        wrong = rng.sample(low, 3)
        options, answer = _shuffle_options(rng, DECADE_NAMES[peak], [DECADE_NAMES[d] for d in wrong])
        questions.append({
            "task_id": 5,
            "question_type": "era_peak",
            "modality": "text",
            "question_text": (
                f"Experts rated the sound of every Billboard Hot 100 hit. In which decade were hits "
                f"most likely to feature {term}?"
            ),
            "audio_reference": "",
            "options": options,
            "correct_answer": answer,
            "reference_explanation": "Average expert rating (0-1) by decade: " + "; ".join(
                f"{DECADE_NAMES[d]} {means[d]:.2f}" for d in decades) + ".",
            "metadata": {"gene": g, "peak_decade": peak, "decade_means": {str(d): round(m, 3) for d, m in means.items()}},
        })
    return questions


# Sound qualities shown in text "era from sound" questions (no lyrics), with plain-language labels.
# Word summaries were too coarse: 147 of 400 songs shared a description, often across decades.
DESCRIBE_GENES = [
    ("Acoustic Sonority", "acoustic sound"), ("Electric Sonority", "electric sound"),
    ("Synthetic Sonority", "electronic sound"), ("Synthetic Drums", "drum machines"),
    ("Drum Set", "drum kit"), ("Drum Aggressiveness", "drum aggressiveness"),
    ("Electric Guitar", "electric guitar"), ("Electric Guitar Distortion", "guitar distortion"),
    ("Acoustic Guitar", "acoustic guitar"), ("Piano", "piano"), ("Organ", "organ"),
    ("Rhodes", "Rhodes electric piano"), ("Synthesizer", "synthesizer"),
    ("String Ensemble", "string section"), ("Horn Ensemble", "horn section"),
    ("Reed Instrument", "saxophone"), ("Percussion", "hand percussion"),
    ("Swing Feel", "swing rhythm"), ("Backbeat", "backbeat"), ("Danceability", "danceability"),
    ("Tempo", "tempo (0 slow, 1 fast)"), ("Minor / Major Key Tonality", "key (0 minor, 1 major)"),
    ("Vocal Grittiness", "vocal grittiness"), ("Vocal Breathiness", "vocal breathiness"),
]


def sound_profile(song_genes: Dict[str, float]) -> Tuple[float, ...]:
    return tuple(round(song_genes[g], 1) for g, _ in DESCRIBE_GENES)


def describe_sound(song_genes: Dict[str, float]) -> str:
    """Expert ratings of one song's sound, without its title, artist or year."""
    return ("Expert ratings of its sound (0 = absent or none, 1 = very strong): "
            + "; ".join(f"{label} {song_genes[g]:.1f}" for g, label in DESCRIBE_GENES) + ".")


def era_from_sound_questions(songs: List[Dict[str, Any]], audio_paths: Dict[str, str],
                             per_decade: int = 50, seed: int = 0) -> List[Dict[str, Any]]:
    """Which decade is this hit from? Text version (expert description) and audio version (clip), same songs.

    Wrong options are at least 20 years away from the right decade, so the question tests the era of a
    sound, not fine dating.
    """
    rng = random.Random(seed + 2)
    groups = by_decade([s for s in songs if s["youtube_id"] in audio_paths])
    decades = sorted(DECADE_NAMES)
    questions = []
    seen = set()  # songs with identical ratings would make the same question with different answers
    for d, rs in groups.items():
        rs = sorted(rs, key=lambda s: s["youtube_id"])
        rng.shuffle(rs)
        unique = []
        for s in rs:
            profile = sound_profile(s["genes"])
            if profile not in seen:
                seen.add(profile)
                unique.append(s)
        for s in unique[:per_decade]:
            far = [x for x in decades if abs(x - d) >= 20]
            wrong = rng.sample(far, 3)
            options, answer = _shuffle_options(rng, DECADE_NAMES[d], [DECADE_NAMES[x] for x in wrong])
            meta = {"song_key": s["mgphot_key"], "youtube_id": s["youtube_id"], "year": s["year"], "decade": d}
            base = {"task_id": 5, "question_type": "era_from_sound", "options": options,
                    "correct_answer": answer,
                    "reference_explanation": f"The song charted in {s['year']}.", "metadata": meta}
            questions.append({**base, "modality": "text", "audio_reference": "", "question_text": (
                "Music experts rated the sound of a Billboard Hot 100 hit. "
                + describe_sound(s["genes"]) + " In which decade was it most likely a hit?")})
            questions.append({**base, "modality": "audio", "audio_reference": audio_paths[s["youtube_id"]],
                              "question_text": "Listen to this 30-second clip of a Billboard Hot 100 hit. "
                                               "In which decade was it most likely a hit?"})
    return questions


# Instruments for the listening-vs-remembering probe: 0 = absent, >= 0.6 = clearly present
PROBE_INSTRUMENTS: Dict[str, str] = {
    "Synthetic Drums": "drum machines",
    "Horn Ensemble": "a horn section",
    "Electric Guitar Distortion": "distorted electric guitar",
    "String Ensemble": "a string section",
    "Piano": "piano",
    "Synthesizer": "synthesizer",
    "Acoustic Guitar": "acoustic guitar",
    "Organ": "organ",
    "Reed Instrument": "saxophone",
    "Rhodes": "Rhodes electric piano",
}


def instrument_probe_questions(songs: List[Dict[str, Any]], audio_paths: Dict[str, str],
                               per_decade: int = 50, seed: int = 0) -> List[Dict[str, Any]]:
    """'Which ONE of these does the recording contain?' asked three ways about the same song.

    audio: the clip only (an audio model must listen); name: title, artist and year (a text model can
    remember the song); era: the year only (what the era alone gives away). Exactly one option is clearly
    present by the experts' ratings and three are absent. The name version shows titles, so these
    questions are a probe set kept out of the benchmark.
    """
    rng = random.Random(seed + 3)
    groups = by_decade([s for s in songs if s["youtube_id"] in audio_paths])
    questions = []
    for d, rs in groups.items():
        rs = sorted(rs, key=lambda s: s["youtube_id"])
        rng.shuffle(rs)
        made = 0
        for s in rs:
            if made >= per_decade:
                break
            g = s["genes"]
            present = [k for k in PROBE_INSTRUMENTS if g[k] >= 0.6]
            absent = [k for k in PROBE_INSTRUMENTS if g[k] == 0.0]
            if not present or len(absent) < 3:
                continue
            right = rng.choice(present)
            wrong = rng.sample(absent, 3)
            options, answer = _shuffle_options(rng, PROBE_INSTRUMENTS[right], [PROBE_INSTRUMENTS[w] for w in wrong])
            ask = "Which ONE of these can be heard in the recording?"
            meta = {"youtube_id": s["youtube_id"], "year": s["year"], "decade": d, "gene": right}
            base = {"task_id": 5, "options": options, "correct_answer": answer, "metadata": meta,
                    "reference_explanation": f"Expert ratings: {right} {g[right]:.1f}; "
                                             + ", ".join(f"{w} 0.0" for w in wrong) + "."}
            questions.append({**base, "question_type": "probe_instrument_audio", "modality": "audio",
                              "audio_reference": audio_paths[s["youtube_id"]],
                              "question_text": f"Listen to this 30-second clip of a Billboard Hot 100 hit. {ask}"})
            questions.append({**base, "question_type": "probe_instrument_name", "modality": "text",
                              "audio_reference": "",
                              "question_text": f'The song "{s["title"]}" by {s["artist"]} was a Billboard Hot 100 '
                                               f"hit in {s['year']}. {ask}"})
            questions.append({**base, "question_type": "probe_instrument_era", "modality": "text",
                              "audio_reference": "",
                              "question_text": f"A song was a Billboard Hot 100 hit in {s['year']}. {ask}"})
            made += 1
    return questions


# MoisesDB instrument categories that are present in some songs and absent in others
MOISESDB_INSTRUMENTS: Dict[str, str] = {
    "piano": "piano",
    "other_keys": "synthesizer or electric keyboard",
    "percussion": "hand percussion",
    "bowed_strings": "bowed strings (violin, cello)",
    "wind": "wind instruments (horns, saxophone, flute)",
    "guitar": "guitar",
    "other_plucked": "other plucked strings (banjo, mandolin, harp)",
}


def unknown_song_probe_questions(tracks: List[Dict[str, Any]], seed: int = 0) -> List[Dict[str, Any]]:
    """The instrument probe on songs no model can have memorised (MoisesDB, mostly unreleased).

    Ground truth is which instrument stems the song was recorded with (MoisesDB ``sources``): present
    means at least one stem of that kind, absent means none. Each track dict needs track_id, title,
    artist, sources and audio_path. Versions: audio (listen) and name (title and artist only).
    """
    rng = random.Random(seed + 4)
    questions = []
    for t in sorted(tracks, key=lambda t: t["track_id"]):
        present = [k for k in MOISESDB_INSTRUMENTS if t["sources"].get(k, 0) > 0]
        absent = [k for k in MOISESDB_INSTRUMENTS if t["sources"].get(k, 0) == 0]
        if not present or len(absent) < 3:
            continue
        # Guitar is in 222 of 240 songs; as the usual answer it would reward always guessing "guitar"
        rarer = [k for k in present if k != "guitar"]
        right = rng.choice(rarer or present)
        wrong = rng.sample(absent, 3)
        options, answer = _shuffle_options(rng, MOISESDB_INSTRUMENTS[right], [MOISESDB_INSTRUMENTS[w] for w in wrong])
        ask = "Which ONE of these can be heard in the recording?"
        base = {"task_id": 5, "options": options, "correct_answer": answer,
                "metadata": {"track_id": t["track_id"], "instrument": right},
                "reference_explanation": f"Recorded stems include {right}; none of: {', '.join(wrong)}."}
        questions.append({**base, "question_type": "probe_unknown_audio", "modality": "audio",
                          "audio_reference": t["audio_path"],
                          "question_text": f"Listen to this recording. {ask}"})
        questions.append({**base, "question_type": "probe_unknown_name", "modality": "text", "audio_reference": "",
                          "question_text": f'The song "{t["title"]}" by {t["artist"]}. {ask}'})
    return questions


def wrong_name_controls(questions: List[Dict[str, Any]], names: Dict[str, Tuple[str, str]],
                        name_type: str, seed: int = 0) -> List[Dict[str, Any]]:
    """Copies of the name questions with another song's title and artist (same decade when known).

    Separates "the model knows this song" from "the title or artist hints at a style": if accuracy
    with a wrong name matches the right name, the gain was not memory of the song.
    names maps a question's song key (youtube_id or track_id) to its (title, artist).
    """
    rng = random.Random(seed + 5)
    src = [q for q in questions if q["question_type"] == name_type]
    key = lambda q: q["metadata"].get("youtube_id") or q["metadata"].get("track_id")
    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for q in src:
        groups.setdefault(q["metadata"].get("decade"), []).append(q)
    out = []
    for qs in groups.values():
        ids = [key(q) for q in qs]
        donors = ids[:]
        for _ in range(100):  # a shuffle where no song keeps its own name
            rng.shuffle(donors)
            if all(a != b for a, b in zip(ids, donors)):
                break
        for q, donor in zip(qs, donors):
            title, artist = names[key(q)]
            d_title, d_artist = names[donor]
            text = q["question_text"].replace(f'"{title}" by {artist}', f'"{d_title}" by {d_artist}', 1)
            out.append({**q, "question_type": name_type.replace("_name", "_wrongname"),
                        "question_text": text, "metadata": {**q["metadata"], "name_from": donor}})
    return out


def why_questions(checklists: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Open-ended 'why did this change?' questions, graded against sourced checklists."""
    questions = []
    for tid, c in sorted(checklists.items(), key=lambda kv: int(kv[0].lstrip("T"))):
        reasons = [r["reason"] for r in c.get("accepted_reasons", []) if r.get("sources")]
        if len(reasons) < 2:
            continue  # not enough sourced reasons to grade fairly
        questions.append({
            "task_id": 5,
            "question_type": "era_why",
            "modality": "text",
            "question_text": (
                f"Among Billboard Hot 100 hits, measurements show this change: {c['trend']} "
                "Why did this happen? Explain the musical, technological, economic or cultural reasons."
            ),
            "audio_reference": "",
            "options": None,
            "correct_answer": None,
            "reference_explanation": " ".join(f"({i + 1}) {r}" for i, r in enumerate(reasons)),
            "metadata": {"trend_id": tid, "checklist": c.get("accepted_reasons", []),
                         "reject": c.get("reject", [])},
        })
    return questions


def option_bias(questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Answer-letter counts and how often the correct option is the single longest (chance ~25%)."""
    mc = [q for q in questions if q.get("options")]
    letters = {k: sum(q["correct_answer"] == k for q in mc) for k in LETTERS}
    longest = 0
    for q in mc:
        lengths = {k: len(v) for k, v in q["options"].items()}
        top = max(lengths.values())
        if lengths[q["correct_answer"]] == top and list(lengths.values()).count(top) == 1:
            longest += 1
    return {"n": len(mc), "letters": letters, "correct_uniquely_longest": longest / max(len(mc), 1)}


def assign_ids(questions: List[Dict[str, Any]], prefix: str = "era") -> List[Dict[str, Any]]:
    import hashlib

    for q in questions:
        song = (q.get("metadata") or {}).get("youtube_id", "")  # songs can share a sound description
        basis = f"{prefix}|{q['question_type']}|{q['modality']}|{song}|{q['question_text']}|{q.get('audio_reference')}"
        q["question_id"] = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:16]
    return questions
