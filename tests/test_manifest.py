"""Tests for musicwhy.data.manifest — track manifest management."""

from __future__ import annotations

from pathlib import Path

import pytest

from musicwhy.data.manifest import TrackManifest, TrackRecord


def _sample_record(**overrides) -> TrackRecord:
    defaults = dict(
        track_id="tr001",
        source="fma",
        file_path="/data/fma/001.mp3",
        title="Test Song",
        artist="Test Artist",
        genre="electronic",
        year=2020,
        duration=180.0,
        license="CC BY 4.0",
        has_stems=False,
        has_features=False,
        has_annotations=False,
    )
    defaults.update(overrides)
    return TrackRecord(**defaults)


def test_track_record_to_dict():
    r = _sample_record()
    d = r.to_dict()
    assert d["track_id"] == "tr001"
    assert d["year"] == 2020
    assert d["has_stems"] is False


def test_track_record_from_dict():
    d = {
        "track_id": "tr002",
        "source": "musdb",
        "file_path": "/data/musdb/002.wav",
        "title": "Song 2",
        "artist": "Artist 2",
        "genre": "rock",
        "year": 2019,
        "duration": 240.5,
        "license": "CC BY-NC 4.0",
        "has_stems": True,
        "has_features": True,
        "has_annotations": False,
    }
    r = TrackRecord.from_dict(d)
    assert r.track_id == "tr002"
    assert r.has_stems is True


def test_manifest_add_and_get(tmp_path: Path):
    manifest = TrackManifest(tmp_path / "manifest.csv")
    r = _sample_record()
    manifest.add_track(r)
    retrieved = manifest.get_track("tr001")
    assert retrieved is not None
    assert retrieved.title == "Test Song"


def test_manifest_duplicate_track(tmp_path: Path):
    manifest = TrackManifest(tmp_path / "manifest.csv")
    r1 = _sample_record()
    r2 = _sample_record(title="Different Title")
    manifest.add_track(r1)
    manifest.add_track(r2)
    assert manifest.get_track("tr001").title == "Different Title"


def test_manifest_filter_by(tmp_path: Path):
    manifest = TrackManifest(tmp_path / "manifest.csv")
    manifest.add_track(_sample_record(track_id="a", genre="rock", year=2020))
    manifest.add_track(_sample_record(track_id="b", genre="rock", year=2021))
    manifest.add_track(_sample_record(track_id="c", genre="jazz", year=2020))
    filtered = manifest.filter_by(genre="rock")
    assert len(filtered) == 2


def test_manifest_stats(tmp_path: Path):
    manifest = TrackManifest(tmp_path / "manifest.csv")
    manifest.add_track(_sample_record(track_id="a", source="fma", duration=100))
    manifest.add_track(_sample_record(track_id="b", source="musdb", duration=200))
    stats = manifest.stats()
    assert stats["total_tracks"] == 2


def test_manifest_track_ids(tmp_path: Path):
    manifest = TrackManifest(tmp_path / "manifest.csv")
    manifest.add_track(_sample_record(track_id="x"))
    manifest.add_track(_sample_record(track_id="y"))
    ids = manifest.track_ids()
    assert set(ids) == {"x", "y"}


def test_manifest_to_dataframe(tmp_path: Path):
    manifest = TrackManifest(tmp_path / "manifest.csv")
    manifest.add_track(_sample_record(track_id="a"))
    manifest.add_track(_sample_record(track_id="b"))
    df = manifest.to_dataframe()
    assert len(df) == 2
    assert "track_id" in df.columns
