"""Phase 0 §4-5: acceptance rates, done criteria and the hand-listening sheet (Test 4.3).

  python -I scripts/phase0_report.py            writes $DATA_ROOT/phase0/phase0_report.md and, once,
                                                 listening_sheet.csv (100 random accepted clips)
Fill the sheet's "matches" column (y/n) for each clip after listening; rerunning adds the false-accept rate.
"""
import csv
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from musicwhy.stimuli import design  # noqa: E402

DATA = Path(os.environ.get("DATA_ROOT", "data"))
OUT = DATA / "phase0"


def rate(k, n):
    return f"{k}/{n} ({100 * k / n:.0f}%)" if n else "0/0"


def main():
    rows = [json.loads(l) for l in (OUT / "clips.jsonl").read_text().splitlines() if l.strip()]
    measured = [r for r in rows if "checks" in r]
    lines = ["# Phase 0 report", "", f"Clips verified: {len(rows)}; measured: {len(measured)}; "
             f"accepted: {rate(sum(r['accepted'] for r in rows), len(rows))}", ""]

    ace_measured = [r for r in measured if not r["checks"].get("midi_by_construction")]
    midi_measured = [r for r in measured if r["checks"].get("midi_by_construction")]
    lines += ["## Acceptance per attribute (ACE-Step clips)", "", "| Attribute | Pass |", "|---|---|"]
    if ace_measured:
        for attr in design.ATTRIBUTES:
            lines.append(f"| {attr} | {rate(sum(r['checks'][attr]['pass'] for r in ace_measured), len(ace_measured))} |")
        octave = sum(r["checks"]["tempo_band"].get("octave_only", False) for r in ace_measured)
        lines += ["", f"Tempo read at half or double the target by both trackers (rejected): {octave}"]
    else:
        lines.append("(no ACE-Step clips in this batch)")
    lines += ["", f"MIDI clips (correct by construction): {rate(sum(r['accepted'] for r in midi_measured), len(midi_measured))}", ""]

    lines += ["## Acceptance by arm", "", "| Arm | Conflicting attribute | Accepted |", "|---|---|---|"]
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[(r["arm"], r["conflict_attr"] or "-")].append(r["accepted"])
    for (arm, attr), acc in sorted(by_arm.items()):
        lines.append(f"| {arm} | {attr} | {rate(sum(acc), len(acc))} |")

    lines += ["", "## Failures", ""]
    reasons = Counter(r.get("reason", "measured") for r in rows if not r["accepted"])
    lines += [f"- {k}: {v}" for k, v in reasons.most_common()]
    planner = sum(bool(r.get("lm_metadata")) for r in rows)
    t32 = [r["test_3_2"]["pass"] for r in rows if r.get("test_3_2")]
    lines += [f"- Test 3.1 (planner metadata present): {planner} clips (must be 0)",
              f"- Test 3.2 (48 kHz, 20 ± 0.5 s, no silence > 2 s): {rate(sum(t32), len(t32))} pass", ""]

    # Done criteria (§5): per song, at least one accepted agree+conflict clip per attribute
    have = defaultdict(set)
    for r in rows:
        if r["accepted"]:
            attr = r.get("test_attr") or r.get("conflict_attr") or "agree"
            have[r["song_id"]].add(attr)
    need = {"agree", *design.ATTRIBUTES}
    complete = sorted(s for s, got in have.items() if need <= got)
    n_songs = len({r["song_id"] for r in rows})
    lines += ["## Done criteria", "", f"Songs with an accepted agree clip and one accepted conflict clip per "
              f"attribute: {rate(len(complete), n_songs)} of songs generated so far (target 500 songs, "
              f"≥ 2,500 accepted clips; accepted now {sum(r['accepted'] for r in rows)})", ""]

    sheet = OUT / "listening_sheet.csv"
    accepted = [r for r in rows if r["accepted"]]
    if not sheet.exists() and len(accepted) >= 100:
        pick = random.Random(0).sample(accepted, 100)
        with open(sheet, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["clip_id", "path", "lead_instrument", "tempo_bpm", "mode", "vocals", "matches", "notes"])
            for r in pick:
                t = r["target"]
                w.writerow([r["clip_id"], r["path"], t["lead_instrument"], t["tempo_bpm"], t["mode"], t["vocals"],
                            "", ""])
        lines.append(f"Listening sheet written: {sheet} (Test 4.3: fill 'matches' with y/n)")
    elif sheet.exists():
        marks = [row["matches"].strip().lower() for row in csv.DictReader(open(sheet, newline=""))]
        done = [m for m in marks if m in ("y", "n")]
        lines.append(f"Test 4.3 false-accept rate: {rate(done.count('n'), len(done))} of {len(marks)} "
                     f"clips marked so far")
    (OUT / "phase0_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
