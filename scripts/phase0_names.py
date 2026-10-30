"""Phase 0 §2: fictional song names with zero corpus hits, joined to the attribute design (§1).

Steps (each resumes; outputs in $DATA_ROOT/phase0/):
  generate  ~2,000 (title, artist) candidates from claude-haiku-5-5   -> names/candidates.jsonl (+ raw replies)
  filter    infini-gram counts on two indexes + MGPHot title overlap  -> names/counts.jsonl, names/survivors.jsonl
  assign    500 survivors + design.assign_attributes                  -> songs.jsonl
  recheck   Test 2.1 on a later day: every song's queries count 0 again -> names/recheck_<date>.jsonl

  python -I scripts/phase0_names.py generate|filter|assign|recheck [--mgphot-dir DIR]
"""
import argparse
import datetime
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from musicwhy.stimuli import design, names  # noqa: E402

DATA = Path(os.environ.get("DATA_ROOT", "data"))
OUT = DATA / "phase0"
NAMES = OUT / "names"
MODEL = os.environ.get("MODEL", "claude-haiku-5-5")
N_CALLS = int(os.environ.get("CALLS", "40"))  # CALLS=2 for the small validated batch first (CLAUDE.md)
PER_CALL, N_SONGS = 50, 500
WORKERS = 4

PROMPT = """Invent {k} titles and artist names for songs that do not exist. Batch {i}: lean toward {theme}.

Rules:
- Each title is 2 to 5 words and combines ordinary words with one rare, unusual or newly coined word, so the
  exact title is unlikely to appear anywhere on the internet.
- Each artist is invented: a coined band name or an unusual first and last name pairing. No real musicians,
  bands, songs, albums, films, books or well-known phrases.
- English. Every title and artist in the batch is different.

Reply with only a JSON array of objects with keys "title" and "artist"."""
THEMES = ["weather and seasons", "cities and travel", "food and kitchens", "machines and tools", "animals",
          "colours and light", "the sea", "night and sleep", "sport and games", "plants and gardens"]


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if path.exists() else []


def append_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def generate():
    import anthropic

    raw_file, cand_file = NAMES / "candidates_raw.jsonl", NAMES / "candidates.jsonl"
    done = {r["call"] for r in read_jsonl(raw_file) if r.get("ok")}
    client = anthropic.Anthropic()

    def call(i):
        prompt = PROMPT.format(k=PER_CALL, i=i, theme=THEMES[i % len(THEMES)])
        for attempt in range(6):
            try:
                # Thinking shares max_tokens; small budgets truncated JSON before (CLAUDE.md)
                resp = client.messages.create(model=MODEL, max_tokens=8000,
                                              messages=[{"role": "user", "content": prompt}])
                break
            except (anthropic.RateLimitError, anthropic.InternalServerError):
                if attempt == 5:
                    raise
                time.sleep(min(60, 5 * 2 ** attempt) + random.random())
        text = "\n".join(b.text for b in resp.content if b.type == "text")
        cands = names.parse_candidates(text) if resp.stop_reason != "max_tokens" and text.strip() else []
        return {"call": i, "model": MODEL, "stop_reason": resp.stop_reason, "raw": text,
                "ok": bool(cands), "n": len(cands)}, cands

    todo = [i for i in range(N_CALLS) if i not in done]
    print(f"{len(todo)} calls to make ({len(done)} done)", flush=True)
    with ThreadPoolExecutor(WORKERS) as pool:
        for row, cands in pool.map(call, todo):
            append_jsonl(raw_file, [row])
            append_jsonl(cand_file, [{**c, "call": row["call"]} for c in cands])
            print(f"  call {row['call']}: {row['n']} candidates, stop_reason={row['stop_reason']}", flush=True)
    cands = read_jsonl(cand_file)
    print(f"{len(cands)} candidates, {len({c['title'].lower() for c in cands})} distinct titles", flush=True)


def filter_(mgphot_dir: str):
    from musicwhy.data.mgphot import load_mgphot

    exact, grams = names.real_title_index(s["title"] for s in load_mgphot(mgphot_dir))
    counts_file = NAMES / "counts.jsonl"
    seen = {(r["title"], r["artist"]) for r in read_jsonl(counts_file)}
    pool_, titles, artists = [], set(), set()
    for c in read_jsonl(NAMES / "candidates.jsonl"):  # first occurrence of each title and each artist
        t, a = c["title"].lower(), c["artist"].lower()
        if t not in titles and a not in artists:
            titles.add(t)
            artists.add(a)
            pool_.append(c)

    def check(c):
        row = {"title": c["title"], "artist": c["artist"], "counts": {}, "checked": str(datetime.date.today())}
        row["mgphot_overlap"] = names.overlaps_real_title(c["title"], exact, grams)
        if row["mgphot_overlap"]:
            return row
        for index in names.INDEXES:  # stop at the first hit: the candidate is out either way
            for q in names.candidate_queries(c["title"], c["artist"]):
                n = names.corpus_count(q, index)
                row["counts"][f"{index}|{q}"] = n
                if n:
                    return row
        return row

    def safe_check(c):
        try:
            return check(c)
        except RuntimeError as e:  # not written, so the next run retries this candidate
            return {"error": str(e)}

    todo = [c for c in pool_ if (c["title"], c["artist"]) not in seen]
    print(f"{len(pool_)} distinct candidates, {len(todo)} to check", flush=True)
    errors = 0
    with ThreadPoolExecutor(2) as pool:  # infini-gram throttles bursts from one address
        for i, row in enumerate(pool.map(safe_check, todo), 1):
            if "error" in row:
                errors += 1
                print(f"  skipped (rerun to retry): {row['error']}", flush=True)
                if errors >= 10:
                    sys.exit("10 candidates failed after retries; the API may be blocking this node. Rerun later.")
                continue
            append_jsonl(counts_file, [row])
            if i % 100 == 0:
                print(f"  {i}/{len(todo)} checked", flush=True)
    rows = read_jsonl(counts_file)
    full = len(names.INDEXES) * 3  # every query was run on every index
    survivors = [r for r in rows if not r["mgphot_overlap"] and len(r["counts"]) == full
                 and not any(r["counts"].values())]
    (NAMES / "survivors.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in survivors),
                                           encoding="utf-8")
    print(f"{len(rows)} checked: {sum(r['mgphot_overlap'] for r in rows)} overlap MGPHot titles, "
          f"{len(survivors)} have zero hits on every query and index", flush=True)


def assign():
    survivors = read_jsonl(NAMES / "survivors.jsonl")
    if len(survivors) < N_SONGS:
        sys.exit(f"only {len(survivors)} survivors; run generate with more calls, then filter again")
    random.Random(0).shuffle(survivors)
    songs = design.assign_attributes(N_SONGS)
    rows = [{"song_id": s["song_id"], "title": n["title"], "artist": n["artist"],
             **{k: s[k] for k in (*design.ATTRIBUTES, "tempo_bpm")}} for s, n in zip(songs, survivors)]
    (OUT / "songs.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                                     encoding="utf-8")
    print(f"{len(rows)} songs -> {OUT / 'songs.jsonl'}", flush=True)


def recheck():
    songs = read_jsonl(OUT / "songs.jsonl")
    today = str(datetime.date.today())

    def check(s):
        counts = {f"{index}|{q}": names.corpus_count(q, index)
                  for index in names.INDEXES for q in names.candidate_queries(s["title"], s["artist"])}
        return {"song_id": s["song_id"], "checked": today, "counts": counts}

    with ThreadPoolExecutor(WORKERS) as pool:
        rows = list(pool.map(check, songs))
    out = NAMES / f"recheck_{today}.jsonl"
    out.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    hits = [r["song_id"] for r in rows if any(r["counts"].values())]
    print(f"Test 2.1 recheck {today}: {len(rows) - len(hits)}/{len(rows)} songs still at zero; hits: {hits[:10]}",
          flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["generate", "filter", "assign", "recheck"])
    p.add_argument("--mgphot-dir", default=str(DATA / "raw" / "mgphot" / "annotations"))
    args = p.parse_args()
    {"generate": generate, "filter": lambda: filter_(args.mgphot_dir), "assign": assign, "recheck": recheck}[args.step]()


if __name__ == "__main__":
    main()
