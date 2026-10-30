"""Blind listening pack: does ACE-Step put the requested instrument in the lead? Runs in ACE-Step's env.

12 instrumental clips (2 seeds x 6 instruments, 108 BPM, major key, renderer only), shuffled and named
pack_01..pack_12 so the listener cannot see what was asked for. The answer key is a separate file.
Writes $DATA_ROOT/phase0/listening_pack/{pack_NN.flac, answer_sheet.csv, key.json}.
"""
import csv
import json
import os
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from musicwhy.stimuli import design  # noqa: E402

OUT = Path(os.environ.get("DATA_ROOT", "data")) / "phase0" / "listening_pack"


def main():
    import acestep.model_downloader as md
    import soundfile as sf
    md.MAIN_MODEL_COMPONENTS[:] = ["acestep-v15-turbo", "vae", "Qwen3-Embedding-0.6B"]
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music

    OUT.mkdir(parents=True, exist_ok=True)
    clips = []
    for inst in design.INSTRUMENTS:
        for k in range(2):
            seed = design.stable_seed("listening_pack", inst, k)
            target = {"lead_instrument": inst, "tempo_band": "medium", "mode": "major", "vocals": "instrumental",
                      "tempo_bpm": 108, "keyscale": design.keyscale("major", seed)}
            clips.append({"seed": seed, "target": target})
    random.Random(0).shuffle(clips)

    dit = AceStepHandler()
    print(dit.initialize_service(project_root=os.environ["ACESTEP_ROOT"], config_path="acestep-v15-turbo",
                                 device="cuda"), flush=True)
    key = {}
    for i, clip in enumerate(clips, 1):
        name = f"pack_{i:02d}"
        result = generate_music(dit, None, GenerationParams(**design.generation_params(clip)),
                                GenerationConfig(batch_size=1, use_random_seed=False, seeds=[clip["seed"]],
                                                 audio_format="wav"), save_dir=str(OUT / "_tmp"))
        audio = result.audios[0]
        sf.write(OUT / f"{name}.flac", audio["tensor"].numpy().T, audio["sample_rate"])
        key[name] = {"requested_lead": clip["target"]["lead_instrument"], "seed": clip["seed"],
                     "caption": design.caption(clip["target"]), "keyscale": clip["target"]["keyscale"]}
    (OUT / "key.json").write_text(json.dumps(key, indent=1))
    with open(OUT / "answer_sheet.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip", "which instrument leads? (piano / electric guitar / saxophone / synthesizer / "
                            "violin / trumpet / other / unclear)", "notes"])
        for name in key:
            w.writerow([name, "", ""])
    print(f"{len(key)} clips -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
