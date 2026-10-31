# Phase 0 v2 — Real stem clips, conflict on the fact side (2026-10-31)
Supersedes the ACE-Step and MIDI clip sources in PHASE0_stimuli_pipeline.md §3–§4. Keeps §1 (attribute idea), §2 (fictional names), tests 1.1, 2.1, 2.2. Drops tests 3.1, 3.2, 4.2, 4.3 (no generator, no human listening).

## Idea in one line
The audio only has to have a *known* answer, not a *generated* one. Real multitrack recordings give the answer by construction. Agree vs conflict is decided by the injected fact, not by the clip: the same clip is used in both.

## Sources (already on Katana)
- MoisesDB, 240 songs, fine-grained stems (Table 2 of arXiv 2307.15913): Piano {Grand, Electric}; Guitar {Acoustic, Clean Electric, Distorted Electric}; Other Keys {Organ, Electric Organ, Synth Lead, Synth Pad, Other}; Bowed Strings {Cello, String Section, Viola…}; Wind {Brass, Reeds, Flutes} (rare); Bass; Drums; Vocals.
- MUSDB18-HQ, 150 songs: vocals/drums/bass/other only → usable for **vocals, tempo, mode**, not instrument.

## Attribute space v2
| Attribute | Values | Ground truth |
|---|---|---|
| `lead_instrument` | piano, electric guitar, acoustic guitar, synthesizer, strings, organ (final set after Step 1 counts; drop any value with < 25 songs) | by construction: sparse mix = drums + bass + exactly one melodic stem group |
| `vocals` | vocal, instrumental | by construction: full mix vs sum of non-vocal stems |
| `tempo_band` | slow 70–85, medium 100–115, fast 130–145 | measured: two trackers agree on the band (existing verifier); else attribute not used for that clip |
| `mode` | major, minor | measured: both Essentia key profiles agree (existing verifier); else not used |

## Steps
1. **Count stems.** From MoisesDB `data.json`, count songs per melodic stem group. Fix the instrument value set. (Test: every kept value ≥ 25 songs.)
2. **Cut windows.** For each song, two non-overlapping 20 s windows where the chosen melodic stem is active ≥ 80% of frames (RMS above −40 dBFS) and vocals (if vocal version) active ≥ 50%. Loudness-normalise all renders to −16 LUFS. (Test: activity thresholds met; sparse-mix + removed stems ≈ full mix within −60 dB.)
3. **Render variants per window.** `sparse_vocal`, `sparse_instrumental` (lead stem + drums + bass ± vocals). Record true attributes: instrument (construction), vocals (construction), tempo/mode (measured; `None` if tools disagree).
4. **Attach fictional songs.** One fictional (title, artist) from §2 per window → ≈480 songs. Assign doses 0/1/10/100/1000 balanced over instrument value (≈96 per dose).
5. **Facts: agree or conflict per attribute, per song.** For each song and each of the 4 attributes, flip a seeded coin: the injected fact equals the clip's true value (agree) or a different value (conflict). Balance so each song has 2 agree + 2 conflict attributes and each attribute is 50/50 across songs. Store `facts.jsonl` {song_id, attr, injected_value, true_value, arm: agree|conflict}.
   - Agree attributes are the control that the model uses audio at all; conflict attributes carry the result.
6. **Dose-0 hearing check (gate for Phase 1).** Qwen2-Audio, no training, on all clips, 2-way for vocals/mode, 4-way for instrument, 3-way for tempo, options rotated. Required: instrument ≥ 65%, vocals ≥ 85%, mode ≥ 65%, tempo ≥ 60%, and the top-letter rate < 40%. If instrument fails, repeat with Audio Flamingo 3; if both fail, the audio arm is not viable with this model class — stop and report.

## Tests
- 1.1 balance (unchanged), 2.1 / 2.2 names (unchanged).
- v2.1 stem counts ≥ 25 per kept instrument value.
- v2.2 every clip: activity and loudness thresholds; reconstruction check.
- v2.3 `facts.jsonl`: 2 agree + 2 conflict per song; 50/50 per attribute; dose groups balanced on instrument.
- v2.4 hearing gate numbers written to `phase0_report.md`.

## Done criteria v2
≥ 400 songs with a clip, true attributes for instrument + vocals on all, tempo + mode on ≥ 70%, facts assigned, hearing gate passed and logged.

## Legal / release
MoisesDB CC BY-NC-SA 4.0 and MUSDB18-HQ non-commercial: local use only. Release: window timestamps, stem recipes, `facts.jsonl`, fictional names, scripts. No audio.

## Do not
- Do not generate any audio. Do not hand-listen. Do not use ACE-Step or MIDI for the main arms (they may stay as an appendix tier if already built).
- Do not pad 2-value attributes to 4 options with fake values.
