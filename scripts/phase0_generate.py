"""Phase 0 §3: generate clips. MIDI clips (instrument/tempo/mode) are rendered locally with tinysoundfont;
ACE-Step 1.5 is used only for clips that need vocals. Runs in ACE-Step's own environment on a GPU node.

The planner (ACE-Step's language model) can never run: its weights are not downloaded, no LLM handler is
created (generate_music skips the LM when llm_handler is None, acestep/inference.py), and thinking and every
CoT option are False. Test 3.1 stops the job if any clip still carries planner metadata.

  ACESTEP_ROOT=... ACESTEP_CHECKPOINTS_DIR=... python scripts/phase0_generate.py --start 0 --count 40 --out-dir DIR
Writes DIR/<clip_id>.wav (48 kHz ACE-Step or 44.1 kHz MIDI) and appends one row per clip to DIR/params.jsonl.
Existing clips are skipped.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from musicwhy.stimuli import design  # noqa: E402
from musicwhy.stimuli.measure import clip_properties  # noqa: E402
from musicwhy.stimuli.render import render_midi, render_params  # noqa: E402

ACE_ROOT = Path(os.environ.get("ACESTEP_ROOT", "/srv/scratch/z5531827/ACE-Step-1.5"))
DIT = "acestep-v15-turbo"
DIT_ONLY_COMPONENTS = ["acestep-v15-turbo", "vae", "Qwen3-Embedding-0.6B"]
SOUNDFONT = Path(os.environ.get("SOUNDFONT", "/srv/scratch/z5531827/.cache/phase0_models/MuseScore_General.sf2"))


def commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(ACE_ROOT), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def _generate_midi(clip, out_dir: Path, duration: float) -> dict:
    """Render a MIDI clip — instrument, tempo, mode exact by construction."""
    spec = render_params(clip)
    path = out_dir / f"{clip['clip_id']}.wav"
    t0 = time.time()
    render_midi(spec, str(SOUNDFONT), path, seconds=duration)
    import soundfile as sf
    data, sr = sf.read(str(path))
    props = clip_properties(data.T, sr)
    return {**clip, "source": "midi", "spec": spec, "generated": True,
            "test_3_2": {**props, "pass": abs(props["duration"] - duration) <= 0.5},
            "path": str(path), "seconds": round(time.time() - t0, 2)}


def _generate_acestep(clip, dit, generate_music, GenerationParams, GenerationConfig,
                      out_dir: Path, duration: float, ace_commit: str) -> dict:
    """Generate a clip with ACE-Step (vocals needed)."""
    kwargs = design.generation_params(clip, duration)
    t0 = time.time()
    result = generate_music(dit, None, GenerationParams(**kwargs),
                            GenerationConfig(batch_size=1, use_random_seed=False, seeds=[clip["seed"]],
                                             audio_format="wav"),
                            save_dir=str(out_dir / "_tmp"))
    if not result.success or not result.audios:
        return {**clip, "source": "acestep", "params": kwargs, "acestep_commit": ace_commit,
                "generated": False, "error": result.error or result.status_message}
    audio = result.audios[0]
    lm_meta = result.extra_outputs.get("lm_metadata")
    if lm_meta:
        sys.exit(f"Test 3.1 FAILED: planner metadata on {clip['clip_id']}: {lm_meta}")
    path = out_dir / f"{clip['clip_id']}.wav"
    shutil.move(audio["path"], path)
    props = clip_properties(audio["tensor"].numpy(), audio["sample_rate"])
    props_ok = (props["sample_rate"] == 48000 and abs(props["duration"] - duration) <= 0.5
                and props["max_silence"] <= 2.0)
    return {**clip, "source": "acestep", "params": kwargs, "acestep_commit": ace_commit,
            "generated": True, "lm_metadata": lm_meta,
            "test_3_2": {**props, "pass": props_ok}, "path": str(path),
            "seconds": round(time.time() - t0, 2)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", type=int, default=0, help="first song index (design order)")
    p.add_argument("--count", type=int, default=40, help="number of songs")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--duration", type=float, default=20.0)
    p.add_argument("--source", choices=["midi", "acestep", "all"], default="all",
                   help="generate only midi or acestep clips (default: all)")
    args = p.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    songs = design.assign_attributes()[args.start:args.start + args.count]
    plan = [c for s in songs for c in design.clip_plan(s)]
    params_file = out / "params.jsonl"
    verified = Path(os.environ.get("DATA_ROOT", "data")) / "phase0" / "clips.jsonl"
    done = set()
    for f in (params_file, verified):
        if f.exists():
            done |= {json.loads(l)["clip_id"] for l in f.read_text().splitlines() if l.strip()}
    todo = [c for c in plan if c["clip_id"] not in done]

    if args.source != "all":
        todo = [c for c in todo if c["source"] == args.source]
    midi_todo = [c for c in todo if c["source"] == "midi"]
    ace_todo = [c for c in todo if c["source"] == "acestep"]
    print(f"songs {args.start}-{args.start + len(songs) - 1}: {len(plan)} clips, "
          f"{len(todo)} to generate ({len(midi_todo)} MIDI, {len(ace_todo)} ACE-Step)", flush=True)

    # --- MIDI clips first (fast, no GPU) ---
    if midi_todo:
        if not SOUNDFONT.exists():
            sys.exit(f"SoundFont not found: {SOUNDFONT}")
        print(f"Rendering {len(midi_todo)} MIDI clips...", flush=True)
        for i, clip in enumerate(midi_todo, 1):
            row = _generate_midi(clip, out, args.duration)
            with open(params_file, "a") as f:
                f.write(json.dumps(row) + "\n")
            if i % 100 == 0 or i == len(midi_todo):
                print(f"  MIDI {i}/{len(midi_todo)}", flush=True)

    # --- ACE-Step clips (GPU, slower) ---
    if ace_todo:
        import acestep.model_downloader as md
        md.MAIN_MODEL_COMPONENTS[:] = DIT_ONLY_COMPONENTS
        from acestep.handler import AceStepHandler
        from acestep.inference import GenerationConfig, GenerationParams, generate_music

        ckpt = Path(os.environ.get("ACESTEP_CHECKPOINTS_DIR", ACE_ROOT / "checkpoints"))
        lm_dirs = sorted(d.name for d in ckpt.glob("acestep-5Hz-lm-*"))
        if lm_dirs:
            sys.exit(f"Test 3.1 setup: planner LM weights present in {ckpt}: {lm_dirs}. Remove them first.")

        dit = AceStepHandler()
        status = dit.initialize_service(project_root=str(ACE_ROOT), config_path=DIT, device="cuda")
        ace_commit = commit()
        print(f"ACE-Step {ace_commit} DiT={DIT}: {status}", flush=True)

        for i, clip in enumerate(ace_todo, 1):
            row = _generate_acestep(clip, dit, generate_music, GenerationParams, GenerationConfig,
                                    out, args.duration, ace_commit)
            with open(params_file, "a") as f:
                f.write(json.dumps(row) + "\n")
            if i % 25 == 0 or i == len(ace_todo):
                print(f"  ACE-Step {i}/{len(ace_todo)}, last took {row.get('seconds', '?')}s", flush=True)
        shutil.rmtree(out / "_tmp", ignore_errors=True)
    else:
        print("No ACE-Step clips needed — skipping GPU model load.", flush=True)


if __name__ == "__main__":
    main()
