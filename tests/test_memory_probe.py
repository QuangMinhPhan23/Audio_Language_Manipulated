import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "memory_probe", Path(__file__).resolve().parents[1] / "scripts" / "memory_probe.py")
mp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mp)


def test_parse_reads_json_with_extra_text():
    text = 'Here you go:\n{"synthetic_drums": true, "horns": "false", "minor_key": false}'
    got = mp.parse(text)
    assert got["synthetic_drums"] is True and got["horns"] is False and got["minor_key"] is False
    assert got["piano"] is None


def test_parse_rejects_missing_json():
    assert mp.parse("I am not sure.") is None


def test_balanced_accuracy_ignores_class_balance():
    # always answering False on 9 false and 1 true fact scores 0.5, not 0.9
    pairs = [(False, False)] * 9 + [(False, True)]
    assert mp.balanced_accuracy(pairs) == 0.5
    assert mp.balanced_accuracy([(True, True), (False, False)]) == 1.0
    assert mp.balanced_accuracy([(True, True)]) is None


def test_lead_artist_matching():
    assert mp.norm("Lil Wayne Featuring Bruno Mars", lead_artist=True) == "lil wayne"
    assert mp.norm("Rock And Roll Is Here To Stay") == "rock and roll is here to stay"
    assert mp.norm("Peter, Paul & Mary", lead_artist=True) == "peter"


def test_name_hidden_outside_name_condition():
    song = {"title": "Secret Title", "artist": "Mystery Band", "year": 1985,
            "measured": {"major_mode": 1.0, "tempo_bpm": 120, "centroid_weighted": 2000, "rms_mean": 0.1,
                         "onset_strength": 1.4, "harmonic_percussive_ratio": 2.0}}
    for cond in ("era", "measured"):
        assert "Secret Title" not in mp.prompt(song, cond) and "Mystery Band" not in mp.prompt(song, cond)
    assert "Secret Title" in mp.prompt(song, "name")
