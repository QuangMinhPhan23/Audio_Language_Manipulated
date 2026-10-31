# Phase 0 review — 2026-10-31
Reviewed commit 0ed2e9e ("Phase 0: fictional songs + verified synthetic clips pipeline"). Results data (`data/`) is not in the repo, so this reviews the code and the decisions recorded in it.

## Verdict
Pipeline is well built (seeded design, every spec test exists in `tests/test_phase0.py`, planner provably off). But one decision made during the build changes the experiment, and it has to be checked before Phase 1 runs:
**instrument / tempo / mode clips are no longer ACE-Step music — they are General-MIDI soundfont loops** (`design.clip_plan`, `render.py`). Only the "vocal" clips still come from ACE-Step.

## What was done well
- `thinking=False`, `llm_handler=None`, planner weights deliberately not downloaded, and Test 3.1 aborts the job if `lm_metadata` ever appears. Stronger than the spec asked.
- Every random choice is seeded from the song id → whole stimulus set is rebuildable byte-for-byte.
- Spec's wrong infini-gram index name was corrected to `v4_olmo-mix-1124_llama` (verified on the API index list today). My error in the spec.
- Tool fallbacks are explicit and logged (beat_this checkpoint server 503 → Essentia; madmom may not import on Py 3.11 → librosa), and the Essentia tempo-doubling quirk found in calibration (72% vs madmom 94% band accuracy) is handled by letting Essentia be at 1× or 2× while the other tracker must match.
- A blind listening pack (12 ACE-Step clips) was built so the human, not the taggers, decides whether ACE-Step obeys instrument captions.

## Problems, in priority order

### P1 — The MIDI pivot was made silently and is not yet justified
Code comments say: "0% of pilot clips passed 'all' with 2 taggers". That is why MIDI was introduced. Two readings are possible and the pipeline has not separated them:
(a) ACE-Step ignores the instrument caption (then MIDI is the right fix), or
(b) ACE-Step obeys but the taggers fail on generated music (then the verifier was the problem, not the generator).
The listening pack answers this. **Action:** listen to the 12 clips, fill `answer_sheet.csv`, run `phase0_tagger_compare.py`. If you hear the requested instrument in ≥ 9/12, reading (b) holds and MIDI should be the secondary arm, not the primary.

### P2 — MIDI clips may be inaudible *to the model*, which would kill the audio arm
The whole paper rests on the model being able to hear the attribute at dose 0. A GM-soundfont I–IV–V–I loop with kick/snare/hi-hat is far from the music Qwen2-Audio was trained on. If dose-0 accuracy on MIDI clips is near chance, "ears lose to memory" is unfalsifiable — the ears never worked.
**Action (before any training):** run `phase0_alm_listen.py` logic over ~200 accepted MIDI clips with Qwen2-Audio, 4-way rotated options, for all three attributes. Required: instrument ≥ 65% (6-way → use the 4-way question), mode ≥ 70%, tempo band ≥ 70%. Below that, the clip source must change (see P3). Record the numbers in `phase0_report.md`; they become the "hearing-intact baseline" row in the paper.

### P3 — Keep a real-music tier so the claim survives review
A reviewer will say "synthetic MIDI toys are not music". The repo already has the answer: 389 true-stem songs (MUSDB18-HQ + MoisesDB). For the instrument attribute, build clips by construction from stems (e.g. keep drums + bass + the one melodic stem that matches the design instrument where the dataset's stem labels allow it). Ground truth is exact, audio is real. Tempo/mode for those clips are measured, not set — acceptable for a secondary tier. Report the dose curve per tier; the headline must hold on at least one real-music tier.

### P4 — Binary attributes padded to 4 options with fake values
`eval_arms.py` pads mode with "chromatic / pentatonic" and vocals with "spoken word / humming". A model can eliminate those trivially, so a "4-way" question on mode is really 2-way and chance is 50%, not 25%. **Action:** ask 2-way for mode and vocals, 4-way for instrument and tempo (tempo needs a 4th band or 3-way). Report chance per attribute. Do not mix chance levels into one accuracy number.

### P5 — Phase 1 training budget does not close
`PHASE1_injection_pipeline.md` §4 notices the 5%-injection rule explodes to ~250k steps and switches to "repeat dose-D songs D times per epoch" — but then the 5% replay ratio is silently dropped. Dose-1000 × 100 songs × 4 attributes = 400k injected examples per epoch; at 5% that is 7.6M replay examples. Not runnable on Katana. **Action:** fix the token budget first: total injected tokens ≈ 400k × ~40 = 16M; choose a replay ratio you can afford (e.g. 1:1, ~32M tokens total, roughly a few hours on one A100 for QLoRA-7B — verify by timing 100 steps) and state it. Chang et al. is the citation for *why* replay, not for the ratio. Also: V100 has no bfloat16 — use fp16 or request an A100 node.

### P6 — Smaller items
- Randomise evidence order in the T arm (evidence-before-question vs after); "Same evidence, different judgments" (2609.26986, Sep 2026) shows order changes the winner in vision/speech-text conflicts. Already noted in the repo's requirements file; make sure `eval_arms.py` actually does it (it currently fixes the prefix position).
- `acceptance.THRESHOLDS["inst_rule"] = "majority"` loosens the spec's rule (target ≥ 0.5 on every tagger). Fine for MIDI (exact by construction, so the tagger is only a sanity check), not fine if ACE-Step instrument clips come back as a tier — restore "all" for those.
- `tests/test_4_2_acceptance_rates_exist` checks the file exists, not the < 30% / < 15% decision rule in the spec. Add the rule.
- The requirements file in the repo was updated on 2026-10-30 with five new papers. Two spot-checked today exist (2609.26986; 2601.06189 = "Rational Synthesizers or Heuristic Followers?"); the specific numbers quoted there (1.3–2.2 docs; 99% / 80% ASR) are **unverified by me** — mark them so until read from the PDFs.

## What I need from Katana to finish this review
1. `data/phase0/calibration/results.json` → the `summary` block (which tempo tools actually ran, band/mode accuracy).
2. `phase0_report.md` or the acceptance counts per attribute and per source (midi vs acestep).
3. Your `answer_sheet.csv` from the listening pack.
4. The dose-0 hearing check from P2 once run.

## Order of work
P1 (one evening, no GPU) → P2 (one GPU job) → decide P3 → P4 + P6 code fixes → P5 budget → Phase 1.
