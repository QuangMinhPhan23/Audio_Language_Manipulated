"""Tests for musicwhy.config — config loading, merging, and validation."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml

from musicwhy.config import (
    AudioConfig,
    BenchmarkConfig,
    EvaluationConfig,
    FeatureConfig,
    MusicWhyConfig,
    ProjectConfig,
    SeparationConfig,
    load_config,
    merge_configs,
)


def test_default_config_loads():
    cfg = load_config(None)
    assert isinstance(cfg, MusicWhyConfig)
    assert cfg.project.seed == 42
    assert cfg.audio.sample_rate == 22050


def test_merge_configs_deep():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"b": 10}, "e": 5}
    merged = merge_configs(base, override)
    assert merged == {"a": {"b": 10, "c": 2}, "d": 3, "e": 5}


def test_merge_configs_empty_override():
    base = {"x": 1}
    assert merge_configs(base, {}) == {"x": 1}


def test_audio_config_defaults():
    cfg = AudioConfig()
    assert cfg.sample_rate == 22050
    assert "wav" in cfg.formats


def test_feature_config_defaults():
    cfg = FeatureConfig()
    assert cfg.mel_bands == 128
    assert cfg.n_mfcc == 20
    assert cfg.hop_lengths == [512, 1024, 2048]


def test_separation_config_defaults():
    cfg = SeparationConfig()
    assert cfg.model == "htdemucs"
    assert cfg.stems == 4


def test_project_config_defaults():
    cfg = ProjectConfig()
    assert cfg.seed == 42
    assert cfg.num_workers == 8


def test_load_config_with_override_file(tmp_path: Path):
    override = tmp_path / "override.yaml"
    override.write_text(
        yaml.dump({"project": {"seed": 99}, "audio": {"sample_rate": 44100}})
    )
    cfg = load_config(override)
    assert cfg.project.seed == 99
    assert cfg.audio.sample_rate == 44100
    assert cfg.features.mel_bands == 128


def test_load_config_with_dict_overrides():
    cfg = load_config(None, overrides={"project": {"seed": 123}})
    assert cfg.project.seed == 123


def test_benchmark_config_tasks():
    cfg = BenchmarkConfig()
    assert "reasoning_qa" in cfg.tasks
    assert "sound_world" in cfg.tasks
    assert cfg.test_ratio > cfg.val_ratio
