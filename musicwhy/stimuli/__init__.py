"""Phase 0 stimuli: fictional songs with known attributes and verified clips.

v1 (ACE-Step/MIDI): design, names, acceptance, measure — see PHASE0_stimuli_pipeline.md.
v2 (real MoisesDB stems): stems, facts — see PHASE0_v2_real_stems.md.

``design``, ``facts`` and ``stems`` (metadata functions) are stdlib-only so they import fast on a cold
node; ``measure`` and ``stems`` (audio functions) load the heavy tools inside their functions.
"""
