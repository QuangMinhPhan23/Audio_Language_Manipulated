# Phase 1 — Memory injection + dose-controlled evaluation

## Goal
Inject fictional song facts into Qwen2-Audio-7B-Instruct via LoRA fine-tuning at five dose levels,
then test whether contradicting audio evidence or text evidence can override the injected memory.
The headline measurement is the T–A gap: the dose at which audio evidence stops working minus the
dose at which text evidence stops working.

## Design (from RESEARCH_REQUIREMENTS.md §7)

- **Model:** Qwen2-Audio-7B-Instruct (Apache-2.0)
- **Songs:** 500 fictional songs from Phase 0, each with 4 known attributes
- **Doses:** 0 / 1 / 10 / 100 / 1000 exposures, 100 songs per group (WITHIN-model: one LoRA
  training run, different songs get different exposure counts)
- **Paraphrases:** ~20 QA templates per attribute = ~80 per song
- **Training:** LoRA on LM attn + FFN only (q/k/v/o/gate/up/down_proj), audio encoder frozen.
  Injected text ≈ 5% of each batch, rest is general replay (Chang recipe).
- **Arms at test time:**
  - **0** (no evidence): ask the question cold
  - **T** (text contradiction): provide a sentence with the wrong attribute value
  - **A** (audio contradiction): provide the Phase 0 conflict clip
- **Metric:** memory-override rate = P(model answers with injected value | contradicting evidence)
- **Hearing-intact check:** MMAU + unknown-song accuracy before/after

---

## Steps

### 1. Paraphrase generation (CPU, local)

For each (song, attribute), produce 20 diverse QA pairs from templates.
Templates vary question phrasing AND answer phrasing, but the factual content is identical.

Output: `data/injection/paraphrases.jsonl` — one row per QA pair.

**Test 1.1:** every song has 80 QA pairs (4 attrs × 20), all paraphrases for the same (song, attr)
agree on the correct answer.

### 2. Dose assignment (CPU, local)

Assign 100 songs to each of 5 dose groups. Seeded, balanced across attribute values
(each dose group is a micro Latin square). Record in `data/injection/dose_groups.json`.

**Test 2.1:** each dose group has exactly 100 songs; attribute value distributions are
within ±3 of the full-population proportions.

### 3. Training data assembly (CPU, local)

For dose D and a song in that group:
- Sample D paraphrases (with replacement when D > 20) from the song's 80 QA pairs.
- Combine all songs' injection examples + general replay at 95:5 ratio.
- Shuffle with a fixed seed.

Replay source: Alpaca-cleaned (52 k general-purpose instruction pairs, CC-BY-NC).

Output: `data/injection/train.jsonl` — single training file with per-example `dose` and `song_id`
metadata (the trainer samples by dose weight).

**Test 3.1:** injection/replay ratio within 4.5–5.5% for a simulated epoch.
**Test 3.2:** no song from dose-0 group appears in injection data.

### 4. LoRA training (GPU, Katana)

One training run. LoRA config:
- r = 32, alpha = 64, dropout = 0.05
- Target modules: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
  (LM layers only; audio encoder + audio projector frozen)
- 4-bit quantization (QLoRA) to fit V100 32 GB
- Learning rate 2e-4, cosine schedule, warmup 100 steps
- Batch size 32 (gradient accumulation as needed)
- Training steps: enough to see every dose-1000 example once ≈ 
  (100 songs × 4 attrs × 1000) / (32 × 0.05) = 250,000 steps.
  This is too many — use epoch-based dosing instead:
  base dataset = all unique paraphrases, repeat dose-D songs D times in the epoch.

**Test 4.1:** after training, dose-0 songs still get ~0% recall (sanity check).
**Test 4.2:** dose-1000 songs get >50% recall on arm-0 (injection worked).
**Test 4.3:** MMAU score within 5% of base model (hearing intact).

### 5. Evaluation (GPU, Katana)

For ALL 500 songs, all 3 arms, all 4 attributes:
- **Arm 0:** "What is the {attribute} of '{title}' by {artist}?" + 4-way MC (rotated)
- **Arm T:** Same question but prefixed with "I recently read that '{title}' by {artist}
  has {wrong_value} as its {attribute}." 
- **Arm A:** Same question but with the Phase 0 conflict clip as audio input.

Output: `data/injection/eval_results.json` — per-question predictions + ground truth.

### 6. Analysis

- Compute memory-override rate per (dose, arm, attribute)
- Plot override rate vs log₁₀(dose) for T and A arms
- The T–A gap is the headline result
- Optional: residual-stream analysis at title tokens

---

## Environment (Katana)

New venv for Phase 1 training:
```bash
python -m venv /srv/scratch/z5531827/venvs/phase1_train
pip install torch transformers peft bitsandbytes accelerate datasets
```

Qwen2-Audio weights: `~/.cache/huggingface/hub/` (re-download needed after cleanup).
