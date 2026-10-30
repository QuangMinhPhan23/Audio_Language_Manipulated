"""MGPHot pilot: a decade-stratified sample of Billboard Hot 100 songs with Music Genome attributes.

Stdlib only. Reads the data files of github.com/MTG/MGPHot-audio (the index without gene values)
and the gene-value TSV on Zenodo record 15372063; none of that repo's code is run.

  python -I scripts/mgphot_pilot.py build  --mgphot-dir D:/WMM_external/MGPHot-audio
  python -I scripts/mgphot_pilot.py check  # is each YouTube video still up? (oEmbed, no download)
  python -I scripts/mgphot_pilot.py download --cookies cookies.txt  # 30 s clips; needs yt-dlp + ffmpeg
  python -I scripts/mgphot_pilot.py compare   # our measurements vs the expert labels

Licence: annotations are non-commercial, and neither they nor the audio may be redistributed.
Keep everything this writes out of the repo and any public upload.
"""
import argparse
import ast
import collections
import csv
import json
import random
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ZENODO_TSV = "https://zenodo.org/records/15372063/files/mgphot_gene_values.tsv?download="
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from musicwhy.data.mgphot import GENES  # noqa: E402


def build(args):
    mg = Path(args.mgphot_dir)
    tsv = mg / "mgphot_gene_values.tsv"
    if not tsv.exists():
        print(f"Downloading {ZENODO_TSV}", flush=True)
        with urllib.request.urlopen(ZENODO_TSV, timeout=120) as r:
            tsv.write_bytes(r.read())
    genes = {}
    with open(tsv, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            vals = [float(x) for x in ast.literal_eval(row["gene_values"])]
            if len(vals) != len(GENES):
                sys.exit(f"{row['artist']} - {row['title']}: {len(vals)} gene values, expected {len(GENES)}")
            genes[(row["artist"].strip(), row["title"].strip())] = vals
    index = json.loads((mg / "genome_index_split_without_gene.json").read_text(encoding="utf-8"))
    songs = []
    for key, rec in index.items():
        g = genes.get((rec["artist"].strip(), rec["title"].strip()))
        if g is not None:
            songs.append({"mgphot_key": key, **rec, "genes": dict(zip(GENES, g))})
    print(f"{len(index)} indexed songs, {len(genes)} gene rows, {len(songs)} joined", flush=True)

    # Same number per decade, official uploads first, so every era is in the pilot
    rng = random.Random(args.seed)
    by_decade = collections.defaultdict(list)
    for s in songs:
        by_decade[s["year"] // 10 * 10].append(s)
    per = args.n // len(by_decade)
    pilot = []
    for decade in sorted(by_decade):
        pool = by_decade[decade]
        rng.shuffle(pool)
        pool.sort(key=lambda s: not s["is_official"])
        pilot += pool[:per]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "pilot_index.json").write_text(json.dumps(pilot, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(pilot)} songs ({per} per decade) to {out / 'pilot_index.json'}")
    print("decades:", dict(collections.Counter(s["year"] // 10 * 10 for s in pilot)))
    print("official uploads:", sum(s["is_official"] for s in pilot))


def check(args):
    """oEmbed answers 200 for a public video and 4xx for removed/private ones. Age or region
    blocks still pass, so this is an upper bound on what download can fetch."""
    out = Path(args.out)
    pilot = json.loads((out / "pilot_index.json").read_text(encoding="utf-8"))
    status = collections.Counter()
    for i, s in enumerate(pilot, 1):
        url = "https://www.youtube.com/oembed?format=json&url=" + urllib.parse.quote(s["youtube_url"], safe="")
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                s["youtube_status"] = r.status
        except urllib.error.HTTPError as e:
            s["youtube_status"] = e.code
        except (urllib.error.URLError, TimeoutError) as e:
            s["youtube_status"] = f"network: {e}"
        status[s["youtube_status"]] += 1
        if i % 25 == 0:
            print(f"  {i}/{len(pilot)} {dict(status)}", flush=True)
        time.sleep(0.3)
    (out / "pilot_index.json").write_text(json.dumps(pilot, indent=1, ensure_ascii=False), encoding="utf-8")
    alive = collections.Counter(s["year"] // 10 * 10 for s in pilot if s["youtube_status"] == 200)
    total = collections.Counter(s["year"] // 10 * 10 for s in pilot)
    print("status:", dict(status))
    print("alive by decade:", {d: f"{alive[d]}/{total[d]}" for d in sorted(total)})


def clip_start(duration, clip):
    """Start 40% of the way in, where a chorus usually is, keeping the whole clip inside the song."""
    return max(0.0, min(duration * 0.4, duration - clip))


def download(args):
    """A FLAC clip per song (what Phase 2 reads), plus a log of every failure and the time taken.

    --clip 30 (default) fetches only that many seconds; --clip 0 fetches the whole song.
    """
    from yt_dlp import YoutubeDL  # pip install yt-dlp; ffmpeg must be on PATH
    from yt_dlp.utils import download_range_func

    out = Path(args.out)
    pilot = json.loads((out / "pilot_index.json").read_text(encoding="utf-8"))
    if args.decades:
        wanted = {int(d) for d in args.decades.split(",")}
        pilot = [s for s in pilot if s["year"] // 10 * 10 in wanted]
        print(f"Only decades {sorted(wanted)}: {len(pilot)} songs", flush=True)
    if args.per_decade:
        taken = collections.Counter()
        keep = []
        for s in pilot:
            decade = s["year"] // 10 * 10
            if taken[decade] < args.per_decade:
                taken[decade] += 1
                keep.append(s)
        pilot = keep
        print(f"Downloading {len(pilot)} songs ({args.per_decade} per decade)", flush=True)
    audio = out / ("audio" if args.clip <= 0 else f"audio_{args.clip:g}s")
    audio.mkdir(exist_ok=True)
    base = {
        "format": "bestaudio/best", "noplaylist": True, "quiet": True, "no_warnings": True,
        "outtmpl": str(audio / "%(id)s.%(ext)s"), "cookiefile": args.cookies,
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "flac"}],
    }
    if args.player_client:
        # Workaround for "The page needs to be reloaded" (yt-dlp issue #17389)
        base["extractor_args"] = {"youtube": {"player_client": args.player_client.split(",")}}
    log = []
    with YoutubeDL(base) as probe:
        for i, s in enumerate(pilot, 1):
            path = audio / f"{s['youtube_id']}.flac"
            if path.exists() or s.get("youtube_status", 200) != 200:
                continue
            t0 = time.time()
            start = end = None
            try:
                opts = dict(base)
                if args.clip > 0:
                    duration = probe.extract_info(s["youtube_url"], download=False).get("duration") or 0
                    start = clip_start(duration, args.clip)
                    end = min(duration, start + args.clip) if duration else args.clip
                    opts["download_ranges"] = download_range_func(None, [(start, end)])
                    opts["force_keyframes_at_cuts"] = True
                with YoutubeDL(opts) as ydl:
                    ydl.download([s["youtube_url"]])
                ok, err = path.exists(), ""
            except Exception as e:  # yt-dlp raises many types; record and move on
                ok, err = False, str(e)[:300]
            log.append({"youtube_id": s["youtube_id"], "year": s["year"], "ok": ok, "error": err,
                        "clip_start": start, "clip_end": end, "seconds": round(time.time() - t0, 1)})
            print(f"[{i}/{len(pilot)}] {s['artist']} - {s['title']} ({s['year']}): "
                  f"{'ok' if ok else 'FAILED ' + err[:80]}", flush=True)
            time.sleep(args.sleep)
    (out / "download_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")
    ok = [r for r in log if r["ok"]]
    print(f"{len(ok)}/{len(log)} downloaded, median {sorted(r['seconds'] for r in ok)[len(ok) // 2] if ok else 0} s/song")


# (our measurement, expert gene) pairs where the two should rise or fall together
COMPARE_PAIRS = [
    ("tempo_bpm", "Tempo"),
    ("major_mode", "Minor / Major Key Tonality"),
    ("harmonic_percussive_ratio", "Acoustic Sonority"),
    ("harmonic_percussive_ratio", "Drum Aggressiveness"),
    ("onset_strength", "Drum Aggressiveness"),
    ("onset_strength", "Danceability"),
    ("rms_mean", "Aural Intensity"),
    ("centroid_weighted", "Synthetic Sonority"),
    ("centroid_weighted", "Electric Guitar Distortion"),
    ("centroid_weighted", "Aural Intensity"),
    ("dynamic_range_db", "Aural Intensity"),
]
OUR_MEASURES = ["tempo_bpm", "centroid_weighted", "rms_mean", "dynamic_range_db", "onset_strength"]
TREND_GENES = ["Synthetic Drums", "Synthetic Sonority", "Electric Guitar Distortion", "Horn Ensemble",
               "Danceability", "Aural Intensity"]


def compare(args):
    """Measure each downloaded clip with the project's feature code and rank-correlate with the experts."""
    import numpy as np
    from scipy.stats import spearmanr

    from musicwhy.features.extract import FeatureExtractor
    from musicwhy.features.feature_to_language import _energy_weighted, estimate_key_mode

    out = Path(args.out)
    pilot = {s["youtube_id"]: s for s in json.loads((out / "pilot_index.json").read_text(encoding="utf-8"))}
    audio = out / f"audio_{args.clip:g}s"
    # Reuse earlier measurements so a rerun only measures new clips
    saved = out / "compare.json"
    cache = {r["youtube_id"]: r for r in json.loads(saved.read_text(encoding="utf-8"))["rows"]
             if "dynamic_range_db" in r} if saved.exists() else {}
    extractor = FeatureExtractor()
    rows = []
    for path in sorted(audio.glob("*.flac")):
        s = pilot.get(path.stem)
        if s is None:
            continue
        if path.stem in cache:
            rows.append(cache[path.stem])
            continue
        if time.time() - path.stat().st_mtime < 120:
            continue  # may still be written by a running download
        try:
            f = extractor.extract_single(path)
        except Exception as e:  # unreadable clip: report and skip
            print(f"  skip {path.name}: {e}", flush=True)
            continue
        mode = estimate_key_mode(f["key_profile"])
        rows.append({
            "youtube_id": path.stem, "year": s["year"], "artist": s["artist"], "title": s["title"],
            "tempo_bpm": f["tempo"],
            "major_mode": None if mode is None else float(mode[1] == "major"),
            "harmonic_percussive_ratio": f["harmonic_percussive_ratio"],
            "onset_strength": float(np.mean(f["onset_strength"])),
            "rms_mean": float(np.mean(f["loudness_contour"])),
            "centroid_weighted": _energy_weighted(f, "spectral_centroid"),
            "dynamic_range_db": f["dynamic_range_db"],
            "genes": s["genes"],
        })
        if len(rows) % 50 == 0:
            print(f"  measured {len(rows)} clips", flush=True)
    print(f"{len(rows)} clips ({len(rows) - len(cache)} newly measured)", flush=True)

    results = []
    tests = COMPARE_PAIRS + [(m, "year") for m in OUR_MEASURES]
    print(f"\nAgreement (Spearman rho); with {len(tests)} tests, p < {0.05 / len(tests):.4f} is solid")
    for ours, gene in tests:
        pairs = [(r[ours], r["year"] if gene == "year" else r["genes"][gene]) for r in rows if r[ours] is not None]
        rho, p = spearmanr([a for a, _ in pairs], [b for _, b in pairs])
        results.append({"ours": ours, "expert": gene, "n": len(pairs), "rho": round(float(rho), 2), "p": float(p)})
        print(f"{ours:>26} vs {gene:<28} n={len(pairs):4d} rho={rho:+.2f} p={p:.4f}")

    by_decade = collections.defaultdict(list)
    for r in rows:
        by_decade[r["year"] // 10 * 10].append(r)
    trends = {}
    print("\nBy decade: our measurements (median) and expert labels (mean, 0-1)")
    cols = OUR_MEASURES + TREND_GENES
    print("decade    n " + " ".join(f"{c[:12]:>12}" for c in cols))
    for d in sorted(by_decade):
        rs = by_decade[d]
        vals = [float(np.median([r[m] for r in rs])) for m in OUR_MEASURES]
        vals += [float(np.mean([r["genes"][g] for r in rs])) for g in TREND_GENES]
        trends[d] = {"n": len(rs), **dict(zip(cols, vals))}
        print(f"{d}s {len(rs):5d} " + " ".join(f"{v:12.3g}" for v in vals))
    saved.write_text(json.dumps({"results": results, "trends": trends, "rows": rows}, indent=1), encoding="utf-8")
    print(f"\nSaved {saved}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("step", choices=["build", "check", "download", "compare"])
    p.add_argument("--mgphot-dir", default="D:/WMM_external/MGPHot-audio")
    p.add_argument("--out", default="D:/WMM_external/mgphot_pilot")
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cookies", default="cookies.txt")
    p.add_argument("--sleep", type=float, default=5.0, help="seconds between downloads")
    p.add_argument("--clip", type=float, default=30.0, help="seconds per song; 0 = whole song")
    p.add_argument("--per-decade", type=int, default=5, help="songs per decade to download; 0 = all 200")
    p.add_argument("--player-client", default="", help="yt-dlp YouTube clients, e.g. default,web_embedded")
    p.add_argument("--decades", default="", help="download only these decades, e.g. 2010,2020")
    args = p.parse_args()
    {"build": build, "check": check, "download": download, "compare": compare}[args.step](args)


if __name__ == "__main__":
    main()
