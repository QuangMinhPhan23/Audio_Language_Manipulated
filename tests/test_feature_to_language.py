"""Tests for musicwhy.features.feature_to_language — feature description generation."""

from __future__ import annotations

import numpy as np
import pytest

from musicwhy.features.feature_to_language import (
    ablation_description,
    features_to_description,
    stem_comparison_description,
)


def _make_features(**overrides) -> dict:
    defaults = dict(
        tempo=120.0,
        spectral_centroid=np.array([2000.0, 2500.0, 2200.0]),
        spectral_bandwidth=np.array([1500.0, 1600.0, 1700.0]),
        spectral_rolloff=np.array([4000.0, 4200.0, 4100.0]),
        harmonic_percussive_ratio=2.5,
        dynamic_range=0.10,
        dynamic_range_db=12.0,
        estimated_key=0,
        key_profile=np.zeros(12),
        mfcc_mean=np.zeros(20),
        duration=210.0,
        onset_strength=np.random.rand(100),
        loudness_contour=np.random.rand(100) * 0.1,
    )
    defaults.update(overrides)
    return defaults


class TestFeaturesToDescription:
    def test_returns_string(self):
        desc = features_to_description(_make_features())
        assert isinstance(desc, str)
        assert len(desc) > 50

    def test_very_slow_tempo(self):
        desc = features_to_description(_make_features(tempo=55.0))
        assert "slow" in desc.lower() or "55" in desc

    def test_fast_tempo(self):
        desc = features_to_description(_make_features(tempo=145.0))
        assert "fast" in desc.lower() or "145" in desc

    def test_very_fast_tempo(self):
        desc = features_to_description(_make_features(tempo=175.0))
        assert "fast" in desc.lower() or "175" in desc

    def test_dark_sound(self):
        desc = features_to_description(
            _make_features(spectral_centroid=np.array([1000.0, 1200.0]))
        )
        assert "dark" in desc.lower() or "warm" in desc.lower()

    def test_bright_sound(self):
        desc = features_to_description(
            _make_features(spectral_centroid=np.array([4000.0, 4500.0]))
        )
        assert "bright" in desc.lower() or "sharp" in desc.lower()

    def test_very_harmonic(self):
        desc = features_to_description(_make_features(harmonic_percussive_ratio=8.0))
        assert "harmonic" in desc.lower() or "tonal" in desc.lower()

    def test_percussive(self):
        desc = features_to_description(_make_features(harmonic_percussive_ratio=0.3))
        assert "percussive" in desc.lower() or "rhythmic" in desc.lower()

    def test_compressed_dynamics(self):
        desc = features_to_description(_make_features(dynamic_range_db=7.0))
        assert "compress" in desc.lower() or "consistent" in desc.lower()

    def test_wide_dynamics(self):
        desc = features_to_description(_make_features(dynamic_range_db=18.0))
        assert "wide" in desc.lower() or "expressive" in desc.lower()

    def test_key_mentioned(self):
        desc = features_to_description(_make_features(estimated_key=9))
        assert "A" in desc


class TestStemComparison:
    def test_returns_string(self):
        full = _make_features()
        stem = _make_features(
            spectral_centroid=np.array([4000.0]),
            harmonic_percussive_ratio=0.5,
        )
        desc = stem_comparison_description(full, stem, "drums")
        assert isinstance(desc, str)


class TestAblationDescription:
    def test_returns_string(self):
        full = _make_features()
        ablated = _make_features(
            spectral_centroid=np.array([1200.0]),
            harmonic_percussive_ratio=6.0,
            dynamic_range_db=7.0,
        )
        desc = ablation_description(full, ablated, "drums")
        assert isinstance(desc, str)
        assert "drums" in desc.lower() or "removed" in desc.lower()


class TestSilentStems:
    def test_silent_stem_is_not_described_as_bright(self):
        from musicwhy.features.feature_to_language import stem_comparison_description

        full = _make_features()
        silent = {"spectral_centroid_mean": 5000.0, "harmonic_percussive_ratio": 1.4, "dynamic_range": 0.0,
                  "rms_mean": 1e-6, "activity_ratio": 0.0}
        desc = stem_comparison_description(full, silent, "bass")
        assert "silent" in desc and "brightness" not in desc


class TestDynamicRangeDb:
    def test_ignores_overall_level(self):
        from musicwhy.features.feature_to_language import dynamic_range_db

        rms = np.linspace(0.01, 0.1, 200)
        assert abs(dynamic_range_db(rms) - dynamic_range_db(rms * 10)) < 1e-9

    def test_ignores_silence(self):
        from musicwhy.features.feature_to_language import dynamic_range_db

        rms = np.linspace(0.01, 0.1, 200)
        assert abs(dynamic_range_db(np.concatenate([np.zeros(300), rms])) - dynamic_range_db(rms)) < 1e-9

    def test_constant_level_is_zero_db(self):
        from musicwhy.features.feature_to_language import dynamic_range_db

        assert dynamic_range_db(np.full(100, 0.05)) == 0.0
