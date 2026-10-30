"""Claim 3 probe: does a text model answer questions about a hit's SOUND from memory of the song?

Same yes/no statements for every song (drum machines? horn section? minor key? ...), asked under:
  era       - only the chart year (what the era alone gives away)
  name      - title, artist and year (memory is possible)
  measured  - year plus our measurements of a 30-second clip, no name
Memory gain = name minus era, compared across famous and lesser hits (peak chart position).
Only facts the Music Genome Project experts are clear about are scored (musicwhy.data.mgphot.clear_facts).

Env: DATA_ROOT, CODE_ROOT, LIMIT (songs; 0 = all), CONDITIONS (default era,name,measured), MODEL.
Input: $DATA_ROOT/raw/mgphot/songs_with_clips.json. Output: $DATA_ROOT/results/memory_probe_<model>.json
(resumes: (song, condition) pairs already answered are skipped).
"""
import json
import os
import random
import re
import sqlite3
import sys
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

CODE_ROOT = Path(os.environ.get("CODE_ROOT") or Path(__file__).resolve().parents[1])
DATA = Path(os.environ.get("DATA_ROOT", "data"))
LIMIT = int(os.environ.get("LIMIT", "0"))
CONDITIONS = os.environ.get("CONDITIONS", "era,name,measured").split(",")
MODEL = os.environ.get("MODEL", "claude-haiku-5-5")
WORKERS = 8
sys.path.insert(0, str(CODE_ROOT))

from musicwhy.data.mgphot import clear_facts  # noqa: E402

ITEMS = {
    "synthetic_drums": ("Synthetic Drums", "The recording uses drum machines or programmed (synthetic) drums."),
    "horns": ("Horn Ensemble", "It has a horn section (trumpets, trombones or saxophones playing together)."),
    "distorted_guitar": ("Electric Guitar Distortion", "It has distorted electric guitar."),
    "strings": ("String Ensemble", "It has a string section (violins, violas, cellos)."),
    "piano": ("Piano", "Piano is clearly audible."),
    "synthesizer": ("Synthesizer", "A synthesizer is clearly audible."),
    "minor_key": ("Minor Key", "The song is in a minor key."),
}


def norm(s, lead_artist=False):
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode().lower()
    if lead_artist:  # "Lil Wayne Featuring Bruno Mars" -> "lil wayne"
        s = re.split(r"\s(?:featuring|feat\.?|ft\.?|with|&|and|x)\s|,", s)[0]
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def chart_history(songs):
    """(peak position, weeks on the Hot 100) per youtube_id, matched on normalised title and lead artist."""
    db = sqlite3.connect(DATA / "world_context" / "world_context.db")
    hist = {}
    for title, artist, peak, weeks in db.execute(
            "SELECT title, artist, MIN(rank), COUNT(*) FROM chart_entries "
            "WHERE source = 'billboard_hot100' AND rank > 0 GROUP BY title, artist"):
        key = (norm(title), norm(artist, lead_artist=True))
        old = hist.get(key)
        if old is None or peak < old[0]:
            hist[key] = (peak, weeks)
    return {s["youtube_id"]: hist.get((norm(s["title"]), norm(s["artist"], lead_artist=True)), (None, None))
            for s in songs}


def context(song, condition):
    if condition == "name":
        return f'The song "{song["title"]}" by {song["artist"]}, a Billboard Hot 100 hit in {song["year"]}.'
    text = f"A song that was a Billboard Hot 100 hit in {song['year']}."
    if condition == "measured":
        m = song["measured"]
        mode = "unknown" if m["major_mode"] is None else ("major" if m["major_mode"] else "minor")
        text += (f" Measurements of a 30-second clip: tempo about {m['tempo_bpm']:.0f} BPM; brightness "
                 f"(energy-weighted spectral centroid) {m['centroid_weighted']:.0f} Hz; loudness (mean RMS) "
                 f"{m['rms_mean']:.3f}; beat strength (mean onset strength) {m['onset_strength']:.2f}; "
                 f"harmonic-to-percussive ratio {m['harmonic_percussive_ratio']:.2f}; detected key mode {mode}.")
    return text


def prompt(song, condition):
    lines = [context(song, condition), "",
             "For each statement about the original recording, answer true or false. Give your best guess "
             "even if you are unsure. Reply with only a JSON object whose keys are the statement names and "
             "whose values are true or false.", ""]
    lines += [f"{k}: {stmt}" for k, (_, stmt) in ITEMS.items()]
    return "\n".join(lines)


def parse(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group())
    except json.JSONDecodeError:
        return None
    out = {}
    for k in ITEMS:
        v = obj.get(k)
        if isinstance(v, str):
            v = {"true": True, "false": False}.get(v.strip().lower())
        out[k] = v if isinstance(v, bool) else None
    return out


def main():
    import anthropic

    songs = json.loads((DATA / "raw" / "mgphot" / "songs_with_clips.json").read_text())
    songs = sorted(songs, key=lambda s: s["youtube_id"])
    if LIMIT:
        songs = songs[:LIMIT]
    fame = chart_history(songs)
    out_file = DATA / "results" / f"memory_probe_{MODEL}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(out_file.read_text())["rows"] if out_file.exists() else []
    done = {(r["youtube_id"], r["condition"]) for r in rows if r.get("answers")}
    rows = [r for r in rows if r.get("answers")]  # failed calls are retried
    todo = [(s, c) for s in songs for c in CONDITIONS if (s["youtube_id"], c) not in done]
    print(f"{len(songs)} songs, {len(todo)} calls to make, {len(rows)} already done; "
          f"chart match for {sum(v[0] is not None for v in fame.values())}/{len(songs)}", flush=True)

    client = anthropic.Anthropic()
    lock = threading.Lock()

    def ask(item):
        song, condition = item
        row = {"youtube_id": song["youtube_id"], "year": song["year"], "condition": condition,
               "peak": fame[song["youtube_id"]][0], "weeks": fame[song["youtube_id"]][1]}
        try:
            for attempt in range(6):
                try:
                    # Thinking shares max_tokens; small budgets truncated JSON before (see CLAUDE.md)
                    resp = client.messages.create(model=MODEL, max_tokens=4000,
                                                  messages=[{"role": "user", "content": prompt(song, condition)}])
                    break
                except (anthropic.RateLimitError, anthropic.InternalServerError):
                    if attempt == 5:
                        raise
                    time.sleep(min(60, 5 * 2 ** attempt) + random.random())  # too many requests at once
            text = "\n".join(b.text for b in resp.content if b.type == "text")
            row["stop_reason"] = resp.stop_reason
            row["raw"] = text
            answers = parse(text) if resp.stop_reason != "max_tokens" and text.strip() else None
            if answers is None:
                row["error"] = f"unparseable or incomplete (stop_reason={resp.stop_reason})"
            row["answers"] = answers
        except Exception as e:  # network or API error: recorded, retried on the next run
            row["error"] = f"{type(e).__name__}: {e}"
            row["answers"] = None
        return row

    with ThreadPoolExecutor(WORKERS) as pool:
        for b in range(0, len(todo), 40):
            batch = list(pool.map(ask, todo[b:b + 40]))
            with lock:
                rows.extend(batch)
                errors = sum(r["answers"] is None for r in batch)
                out_file.write_text(json.dumps({"model": MODEL, "rows": rows}, indent=1))
            print(f"  {min(b + 40, len(todo))}/{len(todo)} calls, {errors} failed in this batch", flush=True)

    summary = summarise(songs, rows)
    out_file.write_text(json.dumps({"model": MODEL, "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1), flush=True)


def balanced_accuracy(pairs):
    """Mean of the hit rate on true facts and on false facts, so 'always false' scores 0.5."""
    pos = [p == t for p, t in pairs if t]
    neg = [p == t for p, t in pairs if not t]
    if not pos or not neg:
        return None
    return round((sum(pos) / len(pos) + sum(neg) / len(neg)) / 2, 3)


def fame_group(peak):
    if peak is None:
        return "no chart match"
    return "top 10" if peak <= 10 else "11-40" if peak <= 40 else "41-100"


def summarise(songs, rows):
    facts = {s["youtube_id"]: clear_facts(s["genes"]) for s in songs}
    by = {}
    for r in rows:
        if not r.get("answers"):
            continue
        f = facts.get(r["youtube_id"], {})
        for k, (gene, _) in ITEMS.items():
            if gene in f and r["answers"].get(k) is not None:
                for group in ("all", fame_group(r["peak"])):
                    by.setdefault((r["condition"], group, k), []).append((r["answers"][k], f[gene]))
    out = {}
    for (cond, group, k), pairs in by.items():
        out.setdefault(cond, {}).setdefault(group, {})[k] = {"n": len(pairs), "bal_acc": balanced_accuracy(pairs)}
    for cond in out:
        for group, items in out[cond].items():
            vals = [v["bal_acc"] for v in items.values() if v["bal_acc"] is not None]
            items["mean"] = round(sum(vals) / len(vals), 3) if vals else None
    return out


if __name__ == "__main__":
    main()
