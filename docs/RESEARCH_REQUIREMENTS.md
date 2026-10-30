# RESEARCH REQUIREMENTS — read at the start of every session
Last updated: 2026-10-23 (second loop pass: "is music just a domain?" — answered, see §4 and §7)

## 1. Non-negotiable goals
- Target: A* CS conference MAIN TRACK (NeurIPS, ICML, ICLR, ACL/EMNLP). ISMIR/ICASSP are fallbacks only, never the goal.
- The contribution must be NEW — not a benchmark twist, not a replication in a new domain. Check §4 before proposing anything.
- No human annotation required. Ground truth by construction, already-labelled (MGPHot), or measurable.
- Legally runnable by one student from Australia: no TDM exception (ruled out 26–27 Oct 2025); fair dealing s40/s103C untested for ML; YouTube download breaches ToS, may be circumvention (s116AN). Never plan on the 1,712 YouTube clips without written UNSW legal clearance.
- Runnable alone on UNSW Katana by ~April 2027 (ISMIR) / ~May 2027 (NeurIPS). ICML Jan 2027 and ACL ARR 4 Jan 2027 are NOT reachable for the injection study.

## 2. How to work (the loop)
1. THINK: idea + strongest version + write the hostile-reviewer paragraph yourself.
2. SEARCH to verify: primary sources only. Find the single most dangerous prior work. Check arXiv ≤ 30 days.
3. If done/wrong/blocked → back to THINK with the precise remaining gap.
4. Exit only when: (a) no prior work does the core thing (URLs), (b) every number sourced this session, (c) legal + solo-feasible, (d) one-sentence headline, (e) method/mechanism/causal — not measurement only.
Never say "novel" without completing step 2 this session.

## 3. Writing rules
Plain language; explain every term once; short; lead with the answer; tables for comparisons; "unverified" over guessing.

## 4. Already published — do NOT re-propose as novel (verified Oct 2026)
- LALMs answer from text priors without audio: All That Glitters (2604.24401), RUListening (ISMIR 2025), MuChoMusic, BASS (2602.04085).
- Typed-text vs audio conflict: ALME (2602.11488 — §2.2 explicitly EXCLUDES parametric memory vs audio), Who Wins (2606.18924), GACL (2606.05161), MCR-Bench (EMNLP 2025). VoxParadox (2605.27772, May 2026): transcript words vs vocal delivery in ONE clip, no memory, evaluation-only.
- Contrastive decoding for audio LLMs: AAD (2506.07233) + 2026 follow-ups.
- Vision "edit a famous thing": VLMs are Biased (2505.23941), POPEv2 (2508.04567), ROHE (NAACL 2025).
- Vision conflict benchmarks on PRE-EXISTING knowledge, no dose: ConflictVis, Koala (2502.14908), WHOOPS-AHA! (Ortu, ACL 2026 long 642), Visual CounterFact, Same Semantics Different Outcome (2609.00550), Vision-Default/Prior-Override (2606.28273: causal heads, 2.5–4.8% of heads flip prior answers 68–96%), Seeing is Believing/ViSiP-Bench (2610.21902, 15 Oct 2026, 24 models).
- **Modality ASYMMETRY in context-memory conflict, vision: "Slow to See, Slow to Suppress" (2609.00293, Brown/Pavlick, 31 Aug 2026).** Pre-existing memory, no dose; the conflicting context is ALWAYS text, the queried entity is text-name vs image. Image-queried entities keep the parametric answer far more (e.g. Gemma-3-12B country: 2.2% text vs 83.0% image). Mechanism: image entities align late → suppression of recall arrives too late. ⇒ "modality changes who wins" is KNOWN for vision. Our gap: evidence channel (not entity channel), audio, and causally controlled memory strength.
- **Injected synthetic facts into a VLM: "Quantifying Cross-Modality Memorization" (2506.05198, Jun 2025).** 100 synthetic personas, LoRA r=32 on Gemma-3-4B, epochs 1–100 as the only exposure knob. Tests RECALL across modalities only; no conflict test, no evidence-vs-memory. ⇒ "LoRA-inject fake entities into a multimodal model" is a known recipe (cite it), not a contribution.
- **Fact editing in an audio LLM: "Localizing and Editing Knowledge in LALMs" (2603.14343, 2026).** Qwen2-Audio-7B-Instruct, CounterFact spoken via TTS, causal tracing + rank-one edits through audio or text; no contradicting-evidence test, no dose. ⇒ Causal tracing in Qwen2-Audio is done; our optional patching step is a tool, not a claim.
- **Dose-controlled memory-vs-TEXT-context: Kim et al., ACL 2026 (2510.02370); Three Regimes (2605.11574).** Audio-only transposition = replication.
- Steering direction for context-vs-memory in text LLMs: 2610.20067 (14 Oct 2026) — a tool to borrow, not a competitor.
- Mechanistic audio-LLM work on EXTERNAL evidence: causal tracing of fusion (2603.13768), audio-specialist heads (2603.06854).
- Generators remember: GEMA v. Suno; Name-Free Gap (2509.00654); Bob's Confetti (2507.17937); Ghost in the Encoder (2609.39552) RQ5 tests ACE-Step planner on/off (0.055/0.022/0.005) and names "song-recognition trigger" as future work.
- ACE-Step 1.5 key obedience: 2608.11899 (exact 0.667). Tempo not tested.
- Memory ∝ frequency in text/image: PopQA, Speak Memory, Xie 2305.13300, Carlini 2301.13188.
- Fictional-fact injection: Chang et al. NeurIPS 2024 (OLMo, batch replacement + replay); Kandpal ICML 2023; Allen-Zhu & Li.
- Unlearning improves grounding: Obliviate (vision).
- STEMMA (2610.11884): stem provenance, no swap, no memory.
- Music Flamingo famous-song remark: 8-song qualitative study only.
- **LoRA on LM layers only preserves audio ability; LoRA on audio tower destroys it: PALM-Bench (2601.03531).** Text-only SFT without replay causes forgetting: ISA-Bench (2510.23558).
- Nothing found (searched 2026-10-23 twice, incl. last 30 days, music AND general audio AND vision): (i) dose-controlled injected memory tested against contradicting PERCEPTUAL evidence in any modality; (ii) same injected facts, text-evidence vs audio-evidence override thresholds in one model.

## 5. Pilot facts (own data, 2026-10-22)
- Text LLM, 400 famous Hot 100 songs: no-info 41%, year-only 54.0%, right title 69.5%, wrong title 43.8%. Title gain over year +15.5; misdirection cost 10.2.
- Decade-majority rule 61% > year-only 54%. Always include this row.
- Within Hot 100, chart peak made NO difference (−0.3, CI −4.1 to +3.5).
- Unknown MoisesDB: right 55%, wrong 57%.
- Qwen2-Audio: instrument 37%/45%, "D" bias 54%; decade-from-sound 64%; stem-attribution 26%.
- Ridge R²: decade 0.15, librosa 0.16, MERT 0.34. Raw features do NOT beat embeddings.
- Assets: 389 true-stem songs (no chart hits), Demucs pipeline, 91 tests.

## 6. Do-not-claim list
- "Naming a song makes generators reproduce it" (GEMA). "First study of ACE-Step's planner" (Ghost). "First ACE-Step key obedience" (2608.11899).
- "First dose-response of memory vs conflicting evidence" (Kim et al. ACL 2026 — text). Claim only the CROSS-MODAL comparison.
- "First to show modality changes conflict resolution" (Slow to See, vision). "First to inject fake entities into a multimodal model" (2506.05198). "First causal tracing in Qwen2-Audio" (2603.14343).
- "Fame increases the effect" (pilot null). "Raw features beat embeddings" (refuted). "MusicGen/SAO never heard the songs."
- "60% of knowledge is recall" (bad denominator). Any legal conclusion about a named company. ICASSP 2027 (closed).
- "This is about music." Frame as audio-language models; music is the first testbed, with a speech arm for generality.

## 7. Current decision state (2026-10-23, pass 2) — LOOP EXITED, design amended
### A* paper (NeurIPS 2027): "Hearing vs reading — evidence through the ears loses to memory at a lower dose than evidence through text"
- Score 4/5 (unchanged). Most dangerous prior now: Slow to See (2609.00293) + Kim et al. (2510.02370) together. Gap: they fix memory and vary where the ENTITY appears (vision); we causally vary memory strength and vary where the EVIDENCE appears (audio). Slow to See's mechanism (late alignment → late suppression) is a prediction we test under dose control, not a finding we repeat.
- Design: Qwen2-Audio-7B-Instruct (Apache-2.0). 500 fictional songs × 4 attributes, ~20 paraphrases each, zero infini-gram hits. Doses 0/1/10/100/1000 (100 songs each). LoRA on LM attn+FFN only, audio encoder frozen, injected text ≈5% of each batch with general replay (Chang recipe; cite 2506.05198 for LoRA-inject-personas precedent). Arms: T (contradicting sentence), A (contradicting ACE-Step 1.5 clip, verified), 0 (no evidence). 4-way option rotation. Hearing-intact check: MMAU + unknown-song accuracy before/after.
- **Amendment (pass 2): add a small SPEECH arm for generality** — 100 fictional people with a voice attribute (pitch class / gender) injected at doses 0/10/1000; conflicting clips made with TTS using VoxParadox's deterministic controls (speaker metadata, SSML pitch). Same T vs A comparison. Purpose: show the dose gap is a property of the audio pathway, not of music. Keep it to ~1/5 the size of the music arm.
- Output: memory-override rate vs log(dose) per arm; the T–A gap is the result. Optional: residual-stream patching at title tokens dose-0 vs dose-1000 (tooling: 2603.14343 already does causal tracing on Qwen2-Audio); test whether 2610.20067's steering direction moves the audio arm. Replicate on AF3 if time.
- Headline: "An ALM stops believing its ears after ~N exposures but keeps believing a contradicting sentence until ~M > N — in music and in speech; the difference is localised to [layer/head]."
- Race: medium-high (Pavlick group could extend Slow to See to audio or to injected facts). Re-run 30-day arXiv search before writing. Post preprint on first clean dose curves.
### Insurance (ISMIR 2027): text-only title-recall study + ACE-Step planner readout (~400 titles, no audio). Preprint if positive.
### Parked: Frankenstein songs (famous arm legally blocked, open models likely can't recognise songs).
