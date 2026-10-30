# Phase 0 — Fictional songs + verified synthetic clips
## Hand this to Claude Code. Every step has a test. Nothing downstream starts until all tests pass.

**Goal:** 500 fictional songs, each with 4 known attributes, and for each a *verified* audio clip that matches the attributes (agree) and one that contradicts one attribute (conflict). All audio from ACE-Step 1.5 (MIT), renderer only, planner OFF, so no song memory leaks into the stimuli.

**Why this first:** clip verification is the slowest, least predictable step (no published tempo/key compliance rates exist for any open generator). Finding the acceptance rate early sets the budget for everything else.

**Build assumptions verified 2026-10-23 (primary sources):**
- ACE-Step 1.5 Python API exposes `bpm`, `keyscale`, `instrumental`, `duration`, `seed`, `thinking=False`, `use_cot_*=False`; output 48 kHz; `extra_outputs["lm_metadata"]` readable — https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md
- infini-gram public API: `POST https://api.infini-gram.io/` with `{"index","query_type":"count","query"}`; OLMo-2 and Dolma v1.7 indexes listed — https://infini-gram.readthedocs.io/en/latest/api.html
- LoRA on language layers only preserves audio ability (PALM-Bench, arXiv 2601.03531) — relevant to Phase 1, not Phase 0.

---

## 0. Environment (Katana)

```bash
pip install acestep  # per https://github.com/ace-step/ACE-Step-1.5 README — needs vllm for LM; we don't use LM but handler init expects it
pip install essentia-tensorflow librosa madmom beat_this requests
# beat_this: https://github.com/CPJKU/beat_this
# Essentia OpenMIC instrument model: https://essentia.upf.edu/models.html
```

Pin versions in `requirements-phase0.txt`. Record ACE-Step commit hash.

---

## 1. Attribute space

| Attribute | Values | Why these |
|---|---|---|
| `lead_instrument` | piano, electric guitar, saxophone, synthesizer, violin, trumpet | 6 classes in OpenMIC vocabulary; acoustically distinct |
| `tempo_band` | slow (70–85), medium (100–115), fast (130–145) BPM | Inside ACE-Step's stable range (docs warn extremes unstable) |
| `mode` | major, minor | Key detector reliability is best on mode, not exact tonic |
| `vocals` | instrumental, vocal | Controlled by `instrumental=True/False`; verify with Demucs vocal-stem energy |

Each song gets one value per attribute, balanced across the 500 (Latin-square style). Store as `songs.jsonl`.

**Test 1.1:** every value appears in 500/n songs ± 5.

---

## 2. Fictional song names

- Generate 2,000 candidate (title, artist) pairs with an LLM, English, plausible, no real-word collisions with famous songs.
- For each, query infini-gram: `POST https://api.infini-gram.io/` with `{"index": "v4_dolma-v1_7_llama", "query_type": "count", "query": "<title>"}` and again with `"<artist>"` and the pair. Keep only pairs where title count == 0 AND artist count == 0.
- Also check the OLMo-2 index (`v4_olmo-2-1124-13b-anneal-adapt_llama`, listed on the API page).
- Keep 500. Save `songs.jsonl` with `{song_id, title, artist, lead_instrument, tempo_bpm, mode, vocals}`.

**Test 2.1:** all 500 titles and artists return count 0 on both indexes (re-run on a fresh day).
**Test 2.2:** no title shares ≥3 consecutive words with any MGPHot title.

---

## 3. Clip generation (ACE-Step 1.5, renderer only)

```python
from acestep.handler import AceStepHandler
from acestep.llm_inference import LLMHandler
from acestep.inference import GenerationParams, GenerationConfig, generate_music

dit = AceStepHandler(); dit.initialize_service(project_root=ROOT, config_path="acestep-v15-turbo", device="cuda")
llm = LLMHandler(); llm.initialize(checkpoint_dir=CKPT, lm_model_path="acestep-5Hz-lm-0.6B", backend="vllm", device="cuda")
# LM is loaded because the handler expects it; thinking=False below means it is never called.

def caption_for(song, conflict_attr=None):
    s = dict(song)
    if conflict_attr: s[conflict_attr] = CONFLICT_VALUE[conflict_attr](s[conflict_attr])
    inst = s["lead_instrument"]
    return f"{inst} lead, {'instrumental' if s['vocals']=='instrumental' else 'with vocals'}, {s['mode']} key, steady beat", s

def make_clip(song, seed, conflict_attr=None):
    cap, s = caption_for(song, conflict_attr)
    params = GenerationParams(
        task_type="text2music",
        caption=cap,
        lyrics="[Instrumental]" if s["vocals"]=="instrumental" else NEUTRAL_LYRICS,
        instrumental=(s["vocals"]=="instrumental"),
        bpm=int(s["tempo_bpm"]),
        keyscale=f"{TONIC[seed % 12]} {'Major' if s['mode']=='major' else 'minor'}",
        timesignature="4",
        duration=20,
        inference_steps=8, shift=3.0, infer_method="ode",
        seed=seed,
        thinking=False,            # planner OFF — renderer only
        use_cot_metas=False, use_cot_caption=False, use_cot_language=False,
    )
    cfg = GenerationConfig(batch_size=1, use_random_seed=False, seeds=[seed], audio_format="wav")
    return generate_music(dit, llm, params, cfg, save_dir=OUT)
```

- Per song: 3 seeds for AGREE + 3 seeds × 4 attributes for CONFLICT = 15 clips/song, 7,500 total. ACE-Step paper reports <2 s/song on A100 — hours, not days.
- `CONFLICT_VALUE`: instrument → a different one of the 6; tempo → a different band; mode → flipped; vocals → flipped.
- Log every `params` dict and the ACE-Step commit.

**Test 3.1:** `thinking=False` → `result.extra_outputs["lm_metadata"]` is empty/None for every clip. If not, the planner ran. Stop.
**Test 3.2:** all clips 48 kHz, 20 ± 0.5 s, no silence > 2 s.

---

## 4. Verification (independent tools, never ACE-Step's own `understand_music`)

| Attribute | Tool A | Tool B | Accept rule |
|---|---|---|---|
| tempo | `beat_this` | `madmom` DBN | both within ±4% of target OR at exact octave (×2, ×0.5) — octave counts as pass for tempo *band* questions only |
| mode | Essentia `KeyExtractor` (edma profile) | Essentia `KeyExtractor` (bgate profile) | both agree on major/minor |
| lead instrument | Essentia OpenMIC model (20 classes) | PaSST (AudioSet head, instrument classes) | target instrument prob ≥ 0.5 on both; every *other* instrument in our 6-class set ≤ 0.2 on both |
| vocals | Demucs `htdemucs` vocal stem RMS vs mix RMS | — | vocal/mix energy ratio < 0.05 for instrumental, > 0.2 for vocal |

A clip is **accepted** only if every attribute passes. For CONFLICT clips, the conflicting attribute must pass *for its conflict value* and the other three for their original values.

**Test 4.1:** run the verifier on 50 known real clips (RWC CC BY-NC subset, or MUSDB18 stems with known tempo/key) → tool agreement ≥ 85% on tempo band and mode. If lower, tune thresholds first.
**Test 4.2:** report acceptance rate per attribute and overall. Expect 40–60% (unverified — no published number). If < 30%, generate more seeds; if < 15%, change the attribute set.
**Test 4.3:** hand-listen to 100 random accepted clips; record false-accept rate. Goes in the paper.

Output: `clips.jsonl` with `{clip_id, song_id, arm: agree|conflict, conflict_attr, seed, path, verifier_scores, accepted}`.

---

## 5. Done criteria for Phase 0

- ≥ 500 songs × ≥ 1 accepted AGREE clip × ≥ 1 accepted CONFLICT clip per attribute (≥ 2,500 accepted clips).
- Tests 1.1, 2.1, 2.2, 3.1, 3.2, 4.1, 4.3 all pass and logged.
- Acceptance rates per attribute written to `phase0_report.md`.

Only then start Phase 1 (fact-text generation + LoRA injection on Qwen2-Audio-7B-Instruct language layers, audio encoder frozen, Chang-style replay).

---

## Do-not-do (from RESEARCH_REQUIREMENTS.md)
- Never verify clips with ACE-Step's own `understand_music` (same model family).
- Never let `thinking=True` touch stimulus generation.
- Never claim a compliance rate for ACE-Step from the literature — measure it.
- No real song titles, lyrics, or artists anywhere in prompts.
