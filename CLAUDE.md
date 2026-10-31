# MusicWhy

**Read `docs/RESEARCH_REQUIREMENTS.md` first.** It holds the current goal (A* main track: dose-controlled injected
memory vs audio evidence), the prior-work list and the do-not-claim list, and it supersedes the research goal
below. Current build step: `docs/PHASE0_v2_real_stems.md` (supersedes §3-§4 of `docs/PHASE0_stimuli_pipeline.md`).
Both files are the canonical specs for the project.

## Research goal (for ARIS skills)

A benchmark and dataset for *why* music sounds the way it does. It covers reasoning from audio features,
stems and world context to emotion, symbolism and cultural moment. (Original goal; superseded by
`RESEARCH_REQUIREMENTS.md`, kept here because the pipeline and checks below were built for it.)

Five tasks: (1) music reasoning QA, (2) source-aware emotion attribution (ground truth = stem-removal
measurements), (3) song relationship explanation, (4) symbolic sound identification, (5) sound-world connection.

Metrics, in priority order:
1. Benchmark validity: `validate.py blind` near 25% (no wording leak); the correct option is not
   systematically the longest (`validate.py benchmark`).
2. Per-task MC accuracy per model (text models on `text` questions, audio models on `audio`), plus the
   model-vs-random and model-vs-human gaps.
3. Open-ended rubric scores (`validate.py eval_open`), with `stop_reason` recorded.
4. Experiment 2 (raw features vs CLAP/MERT embeddings) and Experiment 5 (instrumental vs vocal).

Constraints:
- Don't change `scripts/validate.py` checks or the Task 2 ground truth just to make a number look better.
- Every compute run follows "Running a pipeline stage" below. ARIS experiment skills (`/run-experiment`,
  `/experiment-queue`, `/monitor-experiment`) must use `deploy.sh`, `qsub`, `monitor.py` and the `katana`
  skill, not their own SSH/GPU-server defaults.
- API spend goes through the "Anthropic API calls" rules below. Run one small validated batch before any full run.
- Current best (2026-10-22, benchmark rebuilt with median Task 1 splits and per-task random streams):
  claude-haiku-5-5 text 69% overall (T1 89%, T2 62%, T3 41%, T4 100%); blind (forced guess) 23% (T1 27%,
  T2 16%); open-ended 3.79 / 3.45 / 3.94 (accuracy / depth / specificity). Era benchmark
  (`musicwhy_era_test.json`): text 74% what-changed, 56% peak-decade, 49% era-from-sound; blind 24%;
  why-questions cover 73% of checklist reasons. Qwen2-Audio (audio questions only): main benchmark 27%
  (T1 27%, T2 26%; answers "A" 56% of the time, below an always-one-letter guess of 30-31%); era from
  sound 64%; famous-song instrument probe 37% (answers "D" 54%). Experiment 2 (MGPHot clips, R² over 20
  expert ratings): decade only 0.15, raw measurements 0.16, MERT 0.34. Earlier numbers in
  `data/results_v*` used a parser that misread prose answers.
- Pilot facts and the current plan are in `RESEARCH_REQUIREMENTS.md` §5 and §7. Question files from the pilot
  are on Katana in `data/benchmark/` (`musicwhy_era_test.json`, `musicwhy_probe_test.json`,
  `musicwhy_probe_unknown.json`); raw reviewer exchanges are in `.aris/traces/research-review/`.

## Pipeline

Six-phase benchmark pipeline. Job files are in `jobs/phaseN/*.pbs` and run on Katana at
`/srv/scratch/z5531827/musicwhy`. Cluster basics (queues, walltime, GPU guard) are in the `katana` skill.

## Running a pipeline stage

1. Edit locally, then deploy with `wsl.exe -e bash -lc "bash /mnt/d/WMM/scripts/deploy.sh"`. Don't use ad-hoc scp.
   It strips CRLF, clears `__pycache__`, and syntax-checks every `.pbs`/`.sh`.
2. Before rerunning a stage, check what it will skip. Every stage skips outputs that already exist, so if its
   inputs or code changed, first move the old outputs aside (`mv data/X data/X_<reason>`). Otherwise the
   rerun silently keeps stale results.
3. Submit one stage, then run `wsl.exe -e python3 /mnt/d/WMM/scripts/monitor.py` under the Monitor tool.
   Don't poll with ScheduleWakeup: the monitor prints only state changes, stalls, finished jobs, and
   validation results.
4. The monitor validates the first 5 outputs of per-file stages and every finished job. On `VALIDATE ... FAILED`,
   qdel and fix before spending more compute or API calls.
5. Start the next stage only after `scripts/validate.py <check>` passes. Submit downstream stages by hand.
   Don't build multi-hop `-W depend=afterok` chains here: jobs that bounce change ID, and held jobs did not
   release after their parent had finished.
6. After any qdel, run `qstat -u z5531827` and kill orphaned successors that were already queued.

CPU jobs in every phase (including Phase 1-2 audio jobs, which need librosa/soundfile/pandas) import
code from node-local copies of `bundle/sp_bundle.tar` (third-party packages) and
`bundle/repo_bundle.tar` (musicwhy + scripts) via `mw_stage_bundle` in `_env.sh`. `deploy.sh` rebuilds
only the repo bundle. After adding a dependency to a CPU job, add it to `PKGS` in `scripts/make_bundle.sh`
and run `bash scripts/make_bundle.sh --full` on Katana.

`scripts/validate.py` checks: `manifest world stems annotations benchmark eval eval_open qwen blind pairs all`.
Run it on Katana: `cd /srv/scratch/z5531827/musicwhy && .venv/bin/python -I scripts/validate.py <check>`.

## Katana behaviour seen in this project

- On CPU compute nodes, the first open of each file on scratch can take 0.1-10 s (GPFS, D state, wchan
  `ThCond::internalWait`). Once cached, the same files open in milliseconds, and the login node is
  usually warm. On a cold node `import numpy` took 11-54 s, and the old Phase 3 import tree (964 modules:
  pandas, librosa, scipy, numba) took longer than any watchdog allowed. It isn't a deadlock, so bouncing
  only restarts the cold import. The fix is fewer imported files: `scripts/annotate.py` loads 323 modules
  by avoiding `TrackManifest` (pandas) and `FeatureExtractor` (librosa). `musicwhy.data` is lazy, so
  importing one submodule doesn't pull in the rest.
- `python -I` implies `-E`, which ignores every `PYTHON*` variable. `PYTHONUNBUFFERED` and
  `PYTHONDONTWRITEBYTECODE` in `_env.sh` have no effect on `-I` jobs, so keep `flush=True` on progress prints.
- API-bound jobs (Phases 3 and 5) show near-zero `cput` while working. Judge progress by output count, not CPU.
  `run_evaluation.pbs` makes 8 API calls at a time; one at a time didn't fit the 2 h walltime.
- Many small writes to scratch from a compute node are slow too (FLAC encoding straight to GPFS took
  ~200 s per song vs ~45 s when written to node-local `/tmp` first and copied). The login node's `/tmp`
  is full, so test bundle extraction in a scratch folder instead.
- `PYTHONUNBUFFERED=1` is set in `_env.sh`, so live output is visible:
  `ssh <node> tail /var/spool/pbs/spool/<jobid>.kman.restech.unsw.edu.au.OU`.

## Anthropic API calls

`claude-haiku-5-5` thinks before answering, and thinking shares the `max_tokens` budget. Small budgets
(300, 1500, 2000) gave truncated JSON, empty answers and mid-sentence annotations. That broke the
annotations, the rubric grader and open-ended answers, and all three went unnoticed until validation. For
every call:
- set `max_tokens` to at least 4000 (8000 for long-form);
- join all `type == "text"` blocks instead of taking the first;
- treat `stop_reason == "max_tokens"` or empty text as a failure;
- record `stop_reason` in the output.

## Data conventions

- Sources: 150 MUSDB18-HQ songs (`data/raw/musdb18hq`, WAV) and 239 MoisesDB songs
  (`data/raw/moisesdb/tracks/<uuid>/`, FLAC mixture + 4 stems + `meta.json`; one exact duplicate is
  dropped by `build_manifest`). Genre/licence come from sigsep's `tracklist.csv` and MoisesDB `data.json`.
  Release years (124/389 songs: 95 MUSDB, 29 MoisesDB) come from MusicBrainz/iTunes/Deezer
  (`data/metadata/year_lookup.json`) plus web-search years verified by a second agent
  (`year_websearch.json`, which wins on conflict). The 13 "Music Delta" songs are MedleyDB genre
  pastiches and stay undated; most MoisesDB songs look like library music with no public release.
- Stem features: `data/features/stems/{track_id}_{stem}.json`, computed over the whole song with
  energy-weighted `spectral_centroid_mean` and `activity_ratio`. `stem_comparison_description` reads these
  scalars directly. Stem-removal ("minus one") measurements are in `data/features/ablation/{track_id}.json`
  and are the ground truth for benchmark Task 2.
- Benchmark questions carry `modality` (`text` or `audio`). Text models answer only `text`; run
  `BLIND=1 jobs/phase5/run_evaluation.pbs` after changing question templates and check `validate.py blind`
  stays near 25% (above that, the wording gives the answer away). Blind mode tells the model there is no
  audio and to guess anyway; without that it refused, and refusals are not a leak measurement.
- Results files save every raw reply (`raw_responses`). Unparseable answers are "X" (wrong), never guessed.
- Don't run more than about three API jobs at once: they share the account's concurrent-request limit
  (HTTP 429). Jobs now retry 429s and re-run failed questions when resumed.
- `BENCH=<file> TAG=<suffix>` points the eval jobs at another question file in `data/benchmark/` and keeps
  its results separate (`eval_claude_text<TAG>.json`). MGPHot data stays outside the repo
  (`D:/WMM_external/`, non-commercial, no redistribution); clips on Katana are in `data/raw/mgphot/clips/`.
- `data/features/*.h5` holds one file per `manifest.csv` track (389). The 556 stale files from an old manifest
  that listed stems as tracks were deleted on 2026-10-23 to free scratch space for Phase 0.
- API keys live in `~/.musicwhy_secrets` on Katana (chmod 600), sourced by `_env.sh`. Never put keys in the repo.
- Backups: `data/annotations_old/` (broken metadata), `data/annotations_bad_rms/` (0% stem energy bug),
  `data/annotations_pre_phase1fix/` (empty world context, unknown genres), `data/features/stems_first30s/`,
  `data/embeddings_lastlayer_rawcosine/`, `data/benchmark_v1_heuristic_task2/`,
  `data/benchmark_v2_task1_emotion_leak/`, `data/benchmark_v3_dr_was_loudness/`,
  `data/benchmark_v4_task1_hp_leak/` and matching `data/results_v*` folders (`results_v4_letter_parse_bug/`
  holds runs scored by the old answer parser).
- Dynamic range: use `dynamic_range_db` (dB between loud and quiet non-silent frames). The old
  `dynamic_range` (RMS max minus min) tracked loudness, not dynamics (Spearman 0.79 with mean RMS, 0.04 with
  the dB measure). `load_features` derives `dynamic_range_db` from `loudness_contour` for old .h5 files;
  stem JSONs only get it after `extract_stems.pbs` is rerun.
- Wikipedia year pages are cached in `data/world_context/wiki_cache/`; Wikipedia throttles full-page
  extracts (HTTP 429), so a world-context rerun only fetches pages that are not cached.

## Reporting

Keep status updates to one or two lines. Summarize qstat and log output instead of pasting it.
Explain things simply, in plain words. Avoid jargon and acronyms; when a technical term is unavoidable,
say what it means in a few words the first time.
<!-- ARIS:BEGIN -->
## ARIS Skill Scope
ARIS skills installed in this project: 68 entries.
Manifest: `.aris/installed-skills.txt`
ARIS repo root: `C:\Users\Quang MInh\aris_repo`
Project skill path: `.claude/skills/<skill-name>`
For ARIS workflows, prefer the project-local skills under `.claude/skills/`.
Do not edit or delete junctioned skills in place; update upstream or rerun:
`powershell -NoProfile -ExecutionPolicy Bypass -File "C:\Users\Quang MInh\aris_repo\tools\install_aris.ps1" "D:\WMM" -Platform claude -Reconcile`
<!-- ARIS:END -->
