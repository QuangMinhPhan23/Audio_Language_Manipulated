# Practical feasibility of Idea A ("Frankenstein songs"), Idea B ("generators remember songs by title") and the A+B framing, for a solo UNSW undergraduate on Katana (as of 2026-10-22)

Local project sources used throughout:
- Novelty check: `D:\WMM\Adversarial Novelty Check_ Frankenstein Songs and Generators Remember Songs by Title.md` (cited below as [Novelty check](file:///D:/WMM/Adversarial%20Novelty%20Check_%20Frankenstein%20Songs%20and%20Generators%20Remember%20Songs%20by%20Title.md))
- External review: `D:\WMM\RESEARCH_REVIEW_2026-10-22.md` (cited as [Review 2026-10-22](file:///D:/WMM/RESEARCH_REVIEW_2026-10-22.md))
- Project instructions: `D:\WMM\CLAUDE.md` (cited as [CLAUDE.md](file:///D:/WMM/CLAUDE.md))

Search budget note: about 20 tool calls (searches plus direct fetches of primary pages). Not every sub-question got a primary source. Unsourced points are in the Gaps sections.

---

## Q1. Idea A data: do any clean multitrack datasets contain recognisable (charting) songs? Does the "recognition-gated" design survive?

### Takeaway
No standard research multitrack set (MUSDB18-HQ, MoisesDB, MedleyDB, Cambridge-MT/DSD100, Slakh) contains Billboard-charting *recordings*. The clean, artifact-free arm of Idea A therefore has no songs that a model could recognise, so the "recognized" cell of the recognition-gated design is empty for clean audio. The only routes to recognisable songs are (a) separating commercial recordings, which brings back the YouTube/copyright problem the review flagged, (b) a handful of official artist stem releases (e.g., Nine Inch Nails under CC BY-NC-SA), or (c) Slakh, where the *compositions* come from the Lakh MIDI corpus (many famous songs) but the audio is synthetic and titles are not exposed directly. The recognition-gated design as written collapses to a design with a few songs, or to one that relies on separated commercial audio.

### Cited Findings
- MUSDB18 / MUSDB18-HQ has 150 tracks: 100 from DSD100 (derived from the Cambridge-MT "Mixing Secrets" free multitrack library), 46 from MedleyDB, 2 from Native Instruments' stems pack, and 2 by The Easton Ellises (a "Canadian rock band") from the heise stems remix competition. It is "for academic use only". MedleyDB tracks are CC BY-NC-SA 4.0; the Easton Ellises tracks are CC BY-NC-SA 3.0. Named artists on the page include Al James, Chris Durban, Detsky Sad, Enda Reilly, Hop Along, Leaf and PR. The page does not say any are well known or charted. — [SigSep MUSDB18 page](https://sigsep.github.io/datasets/musdb.html)
- In this project, only 124 of 389 songs (95 MUSDB, 29 MoisesDB) could be dated through MusicBrainz/iTunes/Deezer plus web search. The 13 "Music Delta" songs are MedleyDB genre pastiches, and "most MoisesDB songs look like library music with no public release". — [CLAUDE.md](file:///D:/WMM/CLAUDE.md)
- Slakh2100 is rendered from Lakh MIDI v0.1 with sample-based virtual instruments: 2,100 tracks (1,500/375/225 split), 145 h, at least piano, guitar, drums and bass in every mix. Slakh2100-redux is a deduplicated 1,710-track version. Per-track `metadata.yaml` describes sources. — [Slakh (MERL / Zenodo / slakh-utils)](https://zenodo.org/record/4599666), [mirdata Slakh loader](https://mirdata.readthedocs.io/en/0.3.7/_modules/mirdata/datasets/slakh.html), [slakh-utils README](https://github.com/ethman/slakh-utils/blob/master/README.md)
- Lakh MIDI: LMD-full has 176,581 unique MIDI files named by MD5. LMD-matched has 45,129 files matched to Million Song Dataset entries, with MSD metadata provided for every LMD-matched entry. The "Clean MIDI" subset has artist/title filenames "with some inaccuracy". The LMD page does not say which files Slakh used. — [Lakh MIDI Dataset page](https://colinraffel.com/projects/lmd/)
- Official stem releases: Trent Reznor released the stems of Nine Inch Nails' "The Hand That Feeds" in 2005, and other artists followed. Wikipedia keeps a "List of musical works released in a stem format". — [Wikipedia list](https://en.wikipedia.org/wiki/List_of_musical_works_released_in_a_stem_format)
- Nine Inch Nails' *The Slip* (2008) was released under CC BY-NC-SA 3.0 US, with complete multitracks free at launch. The remix.nin.com site went down in Nov 2016. Remixes are archived on archive.org and the multitracks can still be downloaded from nindestruct.com. Not all NIN multitracks are CC-licensed, so each licence has to be checked. — [Creative Commons case study](https://wiki.creativecommons.org/wiki/Nine_Inch_Nails_The_Slip), [nin.wiki Multitracks](https://nin.wiki/Multitracks), [nin.wiki Remix.nin.com](https://www.nin.wiki/Remix.nin.com)
- The review recommended a text-only paper because famous-song YouTube audio is not legally cleared. MGPHot is CC BY-NC-ND 4.0, and the MGPHot team downloaded audio under an EU research exception that applies to their own institution. — [Review 2026-10-22](file:///D:/WMM/RESEARCH_REVIEW_2026-10-22.md)
- The novelty check's own plan for the high-recognition arm is "separated famous songs (overlap with MGPHot)". — [Novelty check](file:///D:/WMM/Adversarial%20Novelty%20Check_%20Frankenstein%20Songs%20and%20Generators%20Remember%20Songs%20by%20Title.md)

### Inferences
- **Clean arm = unrecognised arm by construction.** Every clean multitrack in hand (MUSDB/MoisesDB/MedleyDB/Cambridge-MT) is indie, student or library music. An open 7B audio model will not identify these songs, and probably neither will a frontier one. So the clean arm can only supply the "unrecognized" control. It cannot test memory override.
- **The "recognized" arm needs famous recordings.** In practice that means separating MGPHot/YouTube audio, which is exactly what the review says is not legally cleared for this researcher. The novelty check's recommended design depends on the one data source the review ruled out.
- **Small legal escape hatches:**
  - NIN official stems (*The Slip* is CC BY-NC-SA; "The Hand That Feeds" stems were officially released). This is a few songs from one artist, a case-study scale, not a dose-response.
  - Native Instruments Stems-format commercial releases. Their licence for research use is unverified.
- **Slakh as a "composition known, recording novel" arm.** Many Lakh MIDI files are transcriptions of famous songs, and Slakh renders them with virtual instruments, so the audio is new but the song is recognisable in principle. This would test whether models recognise a *composition* and then override audio with remembered instrumentation (e.g., the remembered "guitar" versus Slakh's arbitrary patch choices). Song identities must be recovered through LMD-matched/MSD IDs. Whether Slakh keeps the Lakh MD5 IDs needed to look them up was not verified. MIDI renderings are much less recognisable than recordings, so recognition rates would be low.
- **Net assessment.** Without recognisable clean audio, Idea A as a *recognition-gated* study is not feasible for this researcher on a legally clean footing.
  - What remains feasible is a *title-cued* version: clean, unknown multitracks with stems edited, presented with a **false famous title** ("This is 'Billie Jean'") versus no title versus the true (unknown) title.
  - This is a typed-text conflict, which the novelty check's hostile reviewer already calls "a special case of known audio–text conflict" (ALME/GACL).
  - It is feasible, cheap and legal, but its novelty is lower.

### Gaps
- No exact count of MUSDB/MedleyDB artists who ever charted. No source lists chart status, and the project's dataset files are on Katana, not checked locally. Based on the artist names and the project's year lookup, I would expect 0 Billboard Hot 100 entries, but this is unverified.
- Chart peaks of "The Hand That Feeds" and *The Slip* were not verified in this pass.
- Licence terms of Native Instruments Stems releases and of Cambridge-MT multitracks for research redistribution were not checked.
- Whether Australian fair dealing for research (Copyright Act s40) covers separating and editing YouTube audio was not researched. It needs UNSW copyright advice, as the review already said.
- MoisesDB's licence and artist list were not re-checked (CLAUDE.md says the songs look like library music).

---

## Q2. Idea A tools: how good are current separators, what artifacts do stem swaps cause, and how are donor stems key- and tempo-matched?

### Takeaway
State-of-the-art 4-stem separation on MUSDB18-HQ is about 9–12 dB SDR average. "Other" is the weakest stem (7.7–9.0 dB for BS-RoFormer, 6.4 dB for Sparse HT Demucs). Demucs' 6-stem model gives "okay" guitar but "a lot of bleeding and artifacts" on piano. So "removing" or "swapping" a guitar, piano or synth in a separated famous song leaves audible residue in the other stems. A model that still reports the instrument might be *hearing the residue*, not remembering. This confound only goes away with true multitracks.

### Cited Findings
- **Demucs.**
  - HT Demucs reaches 9.00 dB SDR on the MUSDB HQ test set. The fine-tuned HT Demucs v4 is listed at 9.0. A 9.20 dB sparse version is not released.
  - htdemucs_6s has no SDR figure: guitar is "okay", piano has "a lot of bleeding and artifacts" and is "not working great".
  - GPU use: about 7 GB with default arguments, 3 GB minimum with `--segment 8`. The transformer segment is at most 7.8 s. CPU runs at about 1.5× track duration.
  - The repo is archived ("not maintained anymore"). Only important bug fixes go to adefossez/demucs. — [Demucs README](https://github.com/facebookresearch/demucs)
- **BS-RoFormer**, median SDR on MUSDB18-HQ (vocals / bass / drums / other / average):
  - trained on MUSDB18-HQ only (L=6): 10.66 / 11.31 / 9.49 / 7.73 / 9.80;
  - with extra data (L=12): 12.72 / 13.32 / 12.91 / 9.01 / 11.99;
  - Sparse HT Demucs (extra data): 9.37 / 10.47 / 10.83 / 6.41 / 9.27;
  - Spleeter: 5.91 average. — [BS-RoFormer paper (Lu et al., arXiv 2309.02612)](https://ar5iv.labs.arxiv.org/html/2309.02612)
- **Mel-RoFormer** is reported to beat BS-RoFormer on vocals, drums and other. A follow-up table gives vocals 11.49 (BS) vs 12.08 (Mel) trained on MUSDB18-HQ only, and 12.82 vs 13.29 with extra data. The table was extracted from search snippets, so these figures should be checked against the PDFs. — [Mel-Band RoFormer (arXiv 2310.01809)](https://arxiv.org/html/2310.01809v1), [Mel-RoFormer vocal paper (arXiv 2409.04702)](https://arxiv.org/pdf/2409.04702)
- ZFTurbo's Music-Source-Separation-Training repository collects RoFormer checkpoints and training code. It was found but not opened, so its per-model leaderboards are not in these notes. — [ZFTurbo MSST README](https://github.com/ZFTurbo/Music-Source-Separation-Training/blob/016f103397b8ca83fd718aba530ca306485d93dd/README.md?plain=1)
- The novelty check's hostile reviewer predicts the objection "separation artifacts and mismatched key/tempo between donor and host stems make the 'Frankenstein' mixes perceptually unnatural". Its proposed rebuttal is clean multitracks plus key/tempo matching plus a human naturalness check. — [Novelty check](file:///D:/WMM/Adversarial%20Novelty%20Check_%20Frankenstein%20Songs%20and%20Generators%20Remember%20Songs%20by%20Title.md)
- The review flagged that a planned check "altered mix + removed stem = original within −60 dB" cannot pass once the altered mix is loudness-matched. It should report reconstruction error instead. — [Review 2026-10-22](file:///D:/WMM/RESEARCH_REVIEW_2026-10-22.md)

### Inferences
- **Compute is trivial.** Separating 400 songs × about 4 min with a RoFormer or htdemucs on an A100 is well under one 2-hour Katana job. GPU memory (about 7 GB for Demucs) fits any shared GPU.
- **Quality is the problem, not compute.** At about 7–9 dB SDR on "other", residual energy from a "removed" keyboard or guitar is typically still audible in the remaining stems. For Idea A's key contrast (the model claims an instrument that was removed), residue makes "model hallucinated from memory" and "model heard leakage" hard to tell apart. A mitigation is an *energy-based residue check*: measure the removed instrument's leftover energy in the edited mix with a second separator. That second separator has its own errors, though.
- **Swaps are harder than removals.** A donor stem needs key and tempo matching. With stems from the clean, unknown multitracks this is feasible: the project already has per-stem features for 389 songs. Mixing a donor from a different recording also changes room sound and mastering, which is itself a cue.
- **Design choice.** Removal-only edits on clean multitracks avoid both artifacts and key matching. The project's Task 2 stem-removal ("minus-one") pipeline already does exactly this (`data/features/ablation/`), so the infrastructure exists.

### Gaps
- Banquet (query-based separation) SDR numbers were not retrieved.
- SDX23 challenge leaderboard numbers were not retrieved.
- No published measurement of how audible separation residue is to audio-LLMs.
- Key/tempo-matching tools were not verified in this pass: librosa `effects.pitch_shift`/`time_stretch` (phase-vocoder quality) versus pyrubberband (needs the `rubberband` command-line binary on Katana). Whether rubberband is installable without root on Katana is unknown.
- No human-naturalness rating protocol was researched. A listener study would also hit the ethics/Chief-Investigator issue raised in the review.

---

## Q3. Idea A models: can open audio-LLMs identify famous songs from audio, and what do they need to run?

### Takeaway
I found **no published benchmark of song identification (title/artist from audio)** for Qwen2-Audio, Qwen2.5-Omni, Qwen3-Omni, Audio Flamingo 3 or Music Flamingo. The only evidence is Music Flamingo's qualitative remark that *GPT-4o-Audio and Gemini* rely on recognising famous songs. This project's own pilot found Qwen2-Audio at 37% on 4-option famous-song instrument questions, with a strong letter bias. Open 7–8B models fit comfortably on one A100/H100 for clips of 30 s or less. The likely failure is that open models recognise almost nothing, which empties the "recognized" cell.

### Cited Findings
- **No direct benchmark found.** A targeted search for song identification by Qwen2.5-Omni/Gemini found only perception benchmarks:
  - MUSE: instrument identification at 98.33% for Gemini Pro and Qwen.
  - MMAR: Qwen2.5-Omni-7B below 60% average.
  - Core-perception study: Gemini much better with MIDI than with audio. — [MUSE (arXiv 2510.19055)](https://arxiv.org/html/2510.19055v1), [MMAR (arXiv 2505.13032)](https://arxiv.org/pdf/2505.13032), [Core Music Perception (arXiv 2510.22455)](https://arxiv.org/html/2510.22455)
- Music Flamingo's qualitative table says GPT-4o-Audio "relies on recognition of famous songs; may pull from text knowledge instead of audio". This was quoted via the novelty check, which saw the table wording only in a search snippet. — [Novelty check](file:///D:/WMM/Adversarial%20Novelty%20Check_%20Frankenstein%20Songs%20and%20Generators%20Remember%20Songs%20by%20Title.md), [Music Flamingo (arXiv 2511.10289)](https://arxiv.org/html/2511.10289v1)
- **Project pilot.**
  - Qwen2-Audio on famous-song instrument questions: 37%, picking "D" 54% of the time (always-D would score 27%).
  - On unknown songs: 45% (always-D 29%).
  - On the main benchmark: 27%, answering "A" 56% of the time. — [Review 2026-10-22](file:///D:/WMM/RESEARCH_REVIEW_2026-10-22.md), [CLAUDE.md](file:///D:/WMM/CLAUDE.md)
- **Qwen2.5-Omni GPU memory**, theoretical minimum for BF16 with FlashAttention 2 (actual about 1.2× higher):
  - 7B: 31.11 GB (15 s video), 41.85 GB (30 s), 60.19 GB (60 s);
  - 3B: 18.38 / 22.43 / 28.22 GB;
  - 7B GPTQ-Int4: 11.64 / 17.43 / 29.51 GB.

  These figures are for *video* inputs; audio-only needs less (inference). Music scores: MMAU-music 69.16 (7B) vs Qwen2-Audio 50.98; GiantSteps tempo 0.88 (vs LLark 0.86). — [Qwen2.5-Omni GitHub](https://github.com/QwenLM/Qwen2.5-Omni)
- **Qwen2-Audio-7B-Instruct:** 8B parameters, BF16, Apache-2.0. Its model card does not state a maximum audio length or VRAM. — [HF model card](https://huggingface.co/Qwen/Qwen2-Audio-7B-Instruct)
- **Music Flamingo:**
  - 8B, built on a Qwen2.5-7B backbone (Audio Flamingo 3 lineage, AF-Whisper encoder).
  - Processes audio in 30 s windows with a "10-minute total cap per sample". The card also says "Max Audio Length: 20 Minutes", which is inconsistent.
  - Tested on an A100 80 GB; no minimum VRAM stated.
  - **Non-commercial licence** (NVIDIA OneWay Noncommercial).
  - A newer checkpoint, `nvidia/music-flamingo-2601-hf`, exists. — [HF model card](https://huggingface.co/nvidia/music-flamingo-hf)
- **Audio Flamingo 3** supports audio up to 10 minutes (NVIDIA demo page). vLLM processes it in 3,000-frame chunks. — [AF3 HF space](https://nvidia-audio-flamingo-3.hf.space), [vLLM AF3 docs](https://docs.vllm.ai/en/v0.20.2/api/vllm/model_executor/models/audioflamingo3/)
- **Training data teaches models to answer metadata facts as if heard.** Audio Flamingo 2, Music Flamingo and LLark QA-generation prompts feed song metadata (LLark: "artist, song title") and instruct the model to answer "as if… listening". — [Novelty check](file:///D:/WMM/Adversarial%20Novelty%20Check_%20Frankenstein%20Songs%20and%20Generators%20Remember%20Songs%20by%20Title.md)

### Inferences
- **Compute is fine.** On Katana, Qwen2-Audio, Qwen2.5-Omni-7B, AF3 and Music Flamingo all fit one A100/H100 (40–80 GB) in BF16 for clips of 30 s or less. A few thousand items per model fits in 2-hour chunks.
- **The risk is the "recognition" measurement.** The project's 37%-with-letter-bias result suggests Qwen2-Audio barely uses audio for instrument questions, and the open models have no published ability to name songs.
  - If per-song recognition from audio alone is about 0% for open models, the recognition gate and the dose-response cannot be estimated.
  - Only closed APIs (Gemini, GPT-4o-Audio) might recognise songs, and that means uploading copyrighted famous audio to a third-party API, which adds to the legal problem.
- **A cheap go/no-go pilot is available.** Ask each open model "name this song and artist" on about 50 clean MUSDB clips (expected 0%) and about 50 famous clips the researcher can lawfully use, if any (e.g., NIN CC multitracks). If the famous-song recognition rate is below about 10%, drop the recognition gate.
- **Letter bias must be handled first.** Rotate all 4 answer orders and report accuracy averaged over orders. Without this, Idea A's "override" rates will be swamped by position preference.

### Gaps
- Qwen3-Omni's music or song-ID results and memory needs were not retrieved (only the tech report URL surfaced: [Qwen3-Omni Technical Report](https://arxiv.org/pdf/2509.17765)).
- Qwen2-Audio's exact maximum input length was not confirmed from a primary source. It is widely reported as 30 s (Whisper encoder), but unverified here.
- No primary source on VRAM for Audio Flamingo 3 or Music Flamingo beyond "tested on A100 80 GB".
- No published song-identification accuracy for any open audio-LLM.

---

## Q4. Idea B: running ACE-Step 1.5, MusicGen and Stable Audio Open; reading and overriding the planner; measuring tempo, key and instruments; how many generations

### Takeaway
Idea B is **computationally the cheapest and most feasible** part. ACE-Step 1.5 generates a full song in under 2 s on an A100 (0.5–10 s depending on settings). Its planner LMs are 0.6B, 1.7B or 4B Qwen3 models. The planner's output is exposed as `lm_metadata`, and BPM and key can be clamped by passing explicit `bpm`/`keyscale` or `use_cot_metas=False`. So the "planner-clamp" centrepiece is directly supported by the API.

The weak link is *measurement*:
- Beat tracking is good: Beat This reaches 89.1 beat-F1 on GTZAN.
- Key estimation with Essentia has no primary benchmark, and informal tests show about 40–50% accuracy on GiantSteps.
- The review warned that ACE-Step's BPM/key settings are requests, not guarantees.

### Cited Findings
- **ACE-Step 1.5 parameters** (INFERENCE.md):
  - `use_cot_metas` (default True): "Generate metadata using LM CoT reasoning (BPM, key, duration, etc.)".
  - `thinking` (default True): enables the 5 Hz LM chain-of-thought.
  - `bpm` (30–300; `None` enables "auto-detection via LM").
  - `keyscale` (empty string means auto).
  - `duration` (10–600 s).
  - `caption` (max 512 characters).
- **Reading the planner:** `result.extra_outputs["lm_metadata"]` holds "LM-generated metadata" (Example 7 reads `lm_meta.get('bpm')` and `lm_meta.get('keyscale')`). The `cot_*` fields (`cot_bpm`, `cot_keyscale`, `cot_timesignature`, `cot_duration`, `cot_vocal_language`, `cot_caption`, `cot_lyrics`) are "auto-filled by LM".
- **Overriding the planner:** Example 6 sets `use_cot_metas=False`. `batch_size` is 1–8 (default 2), and a VRAM guard reduces it automatically. For GPUs under 20 GB, the docs recommend `offload_to_cpu=True` and INT8. The docs do **not** say how user-set `bpm`/`keyscale` interact with `cot_*` when both are present. — [ACE-Step 1.5 INFERENCE.md](https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md)
- **ACE-Step 1.5 README:**
  - Planner LMs: acestep-5Hz-lm-0.6B / 1.7B / 4B, from Qwen3-0.6B / 1.7B / 4B.
  - VRAM tiers: 0.6B LM about 6–8 GB; 1.7B about 12–16 GB; 4B ≥24 GB (the "20–24GB" row also lists 4B). DiT-only runs fit in ≤6 GB.
  - 2B DiT is about 4.7 GB; an XL 4B DiT is about 9 GB in bf16.
  - Speed: "under 2 seconds per full song on an A100", and "0.5s to 10s on A100 depending on think mode & diffusion steps".
  - Duration 10 s to 600 s. MIT licence.
  - **Training data is not described.** — [ACE-Step 1.5 GitHub](https://github.com/ace-step/ACE-Step-1.5)
- **MusicGen:**
  - Sizes: small 300M, medium 1.5B, melody 1.5B, large 3.3B, melody-large 3.3B; stereo variants exist.
  - "At least 16 GB" GPU for inference with medium.
  - "20K hours of licensed music" (an internal 10K-track set plus Shutterstock and Pond5); training data not released. — [AudioCraft MUSICGEN.md](https://github.com/facebookresearch/audiocraft/blob/main/docs/MUSICGEN.md)
- **Stable Audio Open 1.0:**
  - Stereo 44.1 kHz, up to about 47 s; about 8 GB VRAM (rental-platform guide, not official).
  - A third-party library measured a peak of about 13.8 GB with vanilla Diffusers for 10 s clips on a T4 (bf16); its memory presets brought this to 3.1–5.9 GB.
  - Default 100 diffusion steps. — [Clore.ai guide](https://docs.clore.ai/guides/guides_v2-de/audio-and-stimme/stable-audio), [Audiyo (PyPI)](https://pypi.org/project/audiyo/), [vLLM recipe](https://recipes.vllm.ai/stabilityai/stable-audio-open-1.0)
- **Training data of the controls.** MusicGen trained on licensed audio; Stable Audio Open's music is 13,874 FMA Creative Commons recordings (970 h) with a T5-base text encoder. Any title effect in these models is therefore plausibly text-encoder knowledge, not audio memory. — [Novelty check](file:///D:/WMM/Adversarial%20Novelty%20Check_%20Frankenstein%20Songs%20and%20Generators%20Remember%20Songs%20by%20Title.md) (citing Copet et al. arXiv 2306.05284 and Evans et al. arXiv 2407.14358)
- **Beat This!** GTZAN (test-only): beat F1 89.1 ± 0.3, downbeat F1 78.3 ± 0.4. 8-fold CV beat F1 ranges from 62.7 (SMC) to 99.7 (Candombe); RWC Pop 96.1, Harmonix 95.8. About 20M parameters (a 2M "small" variant also exists). No inference-speed figure. — [Beat This! (arXiv 2407.21658)](https://arxiv.org/html/2407.21658v1)
- **Essentia key:** no MIREX/primary benchmark found. One informal blog test on 40 GiantSteps tracks scored 20/40 (bgate profile) and 16/40 (edma). Essentia's multifeature tempo extractor is reported at 59.9 / 83.2 (Accuracy1 / Accuracy2) on GiantSteps (blog). — summarised via search results; primary links: [MIRFLEX (arXiv 2411.00469)](https://arxiv.org/html/2411.00469), [DEV Community BPM post](https://dev.to/dipak8080/bpm-detection-from-42-to-85-accuracy-with-a-pretrained-model-20al). These are informal sources, so use with caution.
- **The review's warnings on Idea B:**
  - "ACE-Step's tempo and key settings are requests, not guarantees, so 'exact tempo and key' is wrong until each clip is checked."
  - Generated-clip studies need a prompt-leak control. — [Review 2026-10-22](file:///D:/WMM/RESEARCH_REVIEW_2026-10-22.md)
- **Ghost in the Encoder** already analysed ACE-Step 1.5 internals (2,000 songs × 100 artists, lyrics → artist identity). — [Novelty check](file:///D:/WMM/Adversarial%20Novelty%20Check_%20Frankenstein%20Songs%20and%20Generators%20Remember%20Songs%20by%20Title.md)

### Inferences
- **Compute budget (rough).**
  - Design: 400 MGPHot songs × 4 prompt conditions (true title, same-artist wrong title, decade/genre-matched wrong title, no title) × 8 seeds × 30 s clips = 12,800 clips.
  - At 0.5–10 s per clip on an A100, that is about 2–36 GPU-hours, plus Beat This and tagging (small).
  - Repeating with the planner clamped doubles it.
  - MusicGen-large and Stable Audio Open are slower per clip (autoregressive; 100 diffusion steps) but still in the tens of GPU-hours.
  - All of this fits the 2-hour walltime in chunks, and needs no API spend.
- **Planner-only pilot.** The cheapest decisive pilot reads the planner without generating any audio: run the LM with `thinking=True` on about 400 titles and compare `lm_metadata` BPM and key to the true values. The 0.6B and 1.7B planners can do thousands of prompts per hour on one GPU.
  - Caveat: a 0.6B–4B Qwen3 has weak world knowledge, so the planner may *not* know song BPMs. That is a falsifiable, interesting result either way, but if it is null, B's "knowing without hearing" headline weakens.
- **Measurement noise dominates the design.**
  - Tempo: Beat This is reliable, but octave (half/double tempo) errors need folded comparison metrics (inference; standard practice).
  - Key: about 40–50% single-guess accuracy (informal) means key matching must use mir_eval weighted key scores and many seeds, or be dropped as a headline.
  - Instruments: MGPHot's 58 attributes are whole-song expert ratings. Matching them on generated clips needs an instrument tagger, and tagger accuracy (PaSST/OpenMIC) was not retrieved here.
- **How many generations.** No published guidance was found.
  - Paired design: each song is its own control, comparing true title against matched wrong title.
  - Using the pilot's text-model effect (+15.5 points, 95% CI ±about 5.4 with 400 songs and one answer each) as a rough analogue, and given that seeds within a song are correlated, about 5–10 seeds per song per condition with 300–400 songs should give song-level means stable enough for paired tests.
  - Treat this as a heuristic until a 20-song, 20-seed pilot measures between-seed variance.
- **The legal side is easier than A.** Text prompts with titles, and generated outputs, involve no copyrighted input audio. The ground truth needs only MGPHot ratings (licence CC BY-NC-ND, kept outside the repo, per CLAUDE.md). True BPM/key ground truth for Hot 100 songs is the open issue (see Gaps).

### Gaps
- No primary benchmark for Essentia key accuracy (the Essentia docs and MIREX results were not reached).
- Instrument-tagger accuracy (PaSST, OpenMIC mAP; MERT probes) was not retrieved.
- madmom accuracy and maintenance status were not retrieved.
- Source of true tempo/key for Hot 100 songs: whether MGPHot includes tempo was not checked. The Spotify audio-features API is reportedly restricted for new apps since late 2024, but that was not verified here.
- How ACE-Step resolves conflicts between user-set `bpm`/`keyscale` and planner `cot_*` fields is undocumented. It must be tested empirically before the clamp is trusted.
- MusicGen maximum duration and MusicGen-large VRAM were not stated in the docs fetched.
- ACE-Step 1.5 training data (needed for any "renderer heard the song" claim) is undocumented in the sources reached.

---

## Q5. Venues and timeline (as of 2026-10-22)

### Takeaway
- **ICASSP 2027 has already passed** (deadline about 16–23 Sept 2026).
- The next realistic deadlines:
  - **Interspeech 2027:** 9 Feb 2027, a short paper, about 3.5 months away.
  - **ISMIR 2027:** call not yet published. A third-party site says London, 12–16 Sept 2027. Past deadlines suggest about mid/late April 2027.
  - **NeurIPS 2027:** not announced; the 2026 deadline was 6 May 2026, so about early-to-mid May 2027 is likely.
- ICLR 2028 was not verified.

### Cited Findings
- **ICASSP 2027** (Toronto, 16–21 May 2027): the IEEE SPS page lists the paper deadline as 23 Sept 2026; one mldeadlines page says 16 Sept 2026. Either way it is closed. — [IEEE SPS ICASSP 2027](https://signalprocessingsociety.org/events/2027-ieee-international-conference-acoustics-speech-and-signal-processing-icassp), [mldeadlines ICASSP 2027](https://mldeadlines.com/conference/icassp-2027/)
- **Interspeech 2027** (São Paulo, 29 Aug – 2 Sept 2027): paper deadline 9 Feb 2027 23:59 UTC-12, update deadline 16 Feb, notification 2 June 2027. — [mldeadlines Interspeech 2027](https://mldeadlines.com/conference/interspeech-2027/)
- **ISMIR 2027:** no call published. A third-party events site says 12–16 Sept 2027, London (unofficial). ISMIR 2026 had abstracts due 20 Apr 2026 and full papers due 27 Apr 2026, with notification on 10 July. Historical deadlines fall between 12 April and 20 May. — [mldeadlines ISMIR](https://mldeadlines.com/conference/ismir/), [beri.net ISMIR 2027](https://www.beri.net/events/ismir-2027), [ISMIR 2026 CFP posting](https://www.auditory.org/postings/2026/156.html)
- **NeurIPS 2027:** TBA. The 2026 paper deadline was 6 May 2026; one aggregator estimates 16 May 2027 (labelled an estimate). — [underleaf.ai deadlines](https://www.underleaf.ai/conference-deadlines), [trybibby deadlines](https://trybibby.com/conference-deadlines)
- The review's recommended primary venue is ISMIR 2027 (backup TISMIR). Croissant metadata was required for 2026 dataset submissions; the 2027 rules are unknown. — [Review 2026-10-22](file:///D:/WMM/RESEARCH_REVIEW_2026-10-22.md)
- The novelty check suggested an early ICASSP 2027 or Interspeech 2027 short version of A. ICASSP is no longer possible. — [Novelty check](file:///D:/WMM/Adversarial%20Novelty%20Check_%20Frankenstein%20Songs%20and%20Generators%20Remember%20Songs%20by%20Title.md)

### Inferences
- **Realistic sequencing for one person:**
  - an arXiv preprint and/or an Interspeech 2027 short paper (9 Feb) on whichever piece is ready first;
  - ISMIR 2027 (about late April) for the main music paper;
  - NeurIPS 2027 (about May) only if the A+B construct and a mechanistic piece are both done, which is unlikely solo by May.
- **ISMIR and NeurIPS deadlines likely fall 2–4 weeks apart**, so a solo researcher should pick one main target.
- **Interspeech** is speech-focused, but it does take audio-LLM evaluation papers. A short B-pilot (planner-knowledge readout) or a text-only Q paper could fit.

### Gaps
- ICLR 2028 dates were not searched. The historical pattern (late Sept / early Oct abstract deadline) suggests about Sept–Oct 2027, but this is unverified.
- The official ISMIR 2027 site and call could not be confirmed.

---

## Q6. Given the pilot null on popularity within the Hot 100, how risky is the proposed "dose-response on popularity" as a core claim?

### Takeaway
**High risk. It should not be a core claim.** The only direct evidence (text model, 400 Hot 100 songs) shows the title effect is flat across chart bands: top-10 minus 41–100 = −0.3 points, 95% CI −4.1 to +3.5. That interval rules out even a modest 5-point gradient. The reviewer explicitly listed "fame or popularity increases the name effect" as a claim not to make. For audio models, the gradient is likely flat at zero, because open models probably recognise almost no songs. Recognition, measured per item, is a better covariate than popularity, but it is only estimable if some songs are recognised.

### Cited Findings
- **Name gain by chart band:** top-10 +13.6, peak 11–40 +14.4, peak 41–100 +13.9 points. Top-10 minus 41–100 = −0.3 (95% CI −4.1 to +3.5). "The design document's planned headline figure, a famous-to-obscure curve, contradicts this." — [Review 2026-10-22](file:///D:/WMM/RESEARCH_REVIEW_2026-10-22.md)
- **Famous versus unknown songs are not comparable.** The two sets differ in label source, option list, base rates, prompt and clip length. On unknown songs, the right name and a wrong name score the same (55% vs 57%). — [Review 2026-10-22](file:///D:/WMM/RESEARCH_REVIEW_2026-10-22.md)
- **A base rate beats the model given the year.** A no-model decade-prevalence rule scores 61.0%, versus Claude year-only 54.0% and right-name 69.5%. — [Review 2026-10-22](file:///D:/WMM/RESEARCH_REVIEW_2026-10-22.md)
- **The novelty check makes dose-response central.** Its "minimal addition to reach 5" for A is a recognition-gated dose-response with popularity (chart-weeks). Its combined framing requires "a dose-response on song popularity". — [Novelty check](file:///D:/WMM/Adversarial%20Novelty%20Check_%20Frankenstein%20Songs%20and%20Generators%20Remember%20Songs%20by%20Title.md)
- **Reviewer guidance:** "Claims not to make in any version: Fame or popularity increases the name effect." If the fame null replicates, report "chart position does not change the name effect within the Hot 100… no familiarity curve". — [Review 2026-10-22](file:///D:/WMM/RESEARCH_REVIEW_2026-10-22.md)

### Inferences
- **Range restriction is the likely reason for the null.** Every Hot 100 song is famous enough to be well represented in web text, so variation within the Hot 100 may be too small. A real gradient might appear only when extending to non-charting songs. That comparison is confounded, though: the review says the famous and unknown sets differ in labels, base rates and prompts, and an "unknown" song cannot be shown to be unknown to the model.
- **The pre-registered prediction already failed once.** Making the popularity gradient the unifying claim of A+B bets the paper on an effect the only pilot already contradicted. If it fails again, the combined framing loses its "one measurable construct" glue, and the paper reverts to "two workshop papers stapled together" (the novelty check's own phrase).
- **Safer constructs:**
  1. *Item-level knowledge* measured directly: does the text or planner model know the song's attributes from the title, as a per-song text-only score? Use it as the moderator of audio override (A) and planner drift (B). This is a "knowledge-gated" rather than a "popularity" dose-response, and it is measurable for every model.
  2. Report popularity only as a secondary, pre-registered test, with the existing null as the expected outcome.
- **For B, the gradient is cheap to test.** Reading ACE-Step planner BPM/key error by chart band needs no audio and only a few GPU-hours. It could be run before committing to the framing.

### Gaps
- No evidence exists on whether a popularity gradient appears *beyond* the Hot 100 under matched labels. No such matched set exists in the project.
- No evidence on audio-model recognition rates by chart band. This is a direct consequence of the missing song-ID benchmarks in Q3.

---

## Overall feasibility summary (inference, for the report writer)

| Component | Compute | Data / legal | Main failure risk | Feasibility (solo, by May 2027) |
|---|---|---|---|---|
| A, clean-multitrack arm (unknown songs, stem removal/swap, false famous title) | Trivial (inference only; existing minus-one pipeline) | Clean (MUSDB/MoisesDB academic licences) | Reduces to typed-text conflict (ALME/GACL territory). Letter bias in Qwen2-Audio. | High, but lower novelty |
| A, recognition-gated famous-song arm | Small (separation under 1 GPU-h; audio-LLM inference tens of GPU-h) | **Blocked**: needs separated commercial audio (YouTube/MGPHot clips not cleared); only a few NIN CC multitracks are clean | Open models may recognise about 0 songs. Separation residue (7–9 dB SDR on "other"; Demucs piano bleed) confounds "hallucination". | Low |
| B, ACE-Step planner readout + clamp | Small (<2–10 s per clip on A100; planner-only pilot needs no audio) | Clean (text prompts; MGPHot ratings non-commercial, kept outside the repo) | Small Qwen3 planners may not know song BPM/key. Key estimation noisy. `cot_*` vs user-field interaction undocumented. Ghost-in-the-Encoder group is adjacent. | Medium-high |
| B controls (MusicGen, Stable Audio Open) | Tens of GPU-hours | Clean | Effect attributable to T5/text-encoder knowledge (the reviewer's objection) | High (as controls) |
| Combined A+B with popularity dose-response | Sum of the above | A's famous arm blocked | Pilot null on popularity (−0.3 pts, CI −4.1 to +3.5) | Low as framed. Medium if reframed around measured per-song knowledge instead of popularity. |
