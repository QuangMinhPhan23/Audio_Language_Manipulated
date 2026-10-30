"""Pydantic-based configuration system for MusicWhy."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Default config location
# ---------------------------------------------------------------------------
_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


# ---------------------------------------------------------------------------
# Section models
# ---------------------------------------------------------------------------
class AudioConfig(BaseModel):
    """Audio loading and preprocessing settings."""

    sample_rate: int = Field(default=22050, ge=8000, le=48000)
    max_duration: int = Field(default=600, ge=1, description="Max duration in seconds")
    formats: list[str] = Field(default_factory=lambda: ["wav", "mp3", "flac", "ogg"])


class FeatureConfig(BaseModel):
    """Audio feature extraction settings."""

    mel_bands: int = Field(default=128, ge=20, le=512)
    hop_lengths: list[int] = Field(default_factory=lambda: [512, 1024, 2048])
    n_mfcc: int = Field(default=20, ge=1, le=40)
    store_format: str = Field(default="hdf5")


class SeparationConfig(BaseModel):
    """Source separation settings."""

    model: str = Field(default="htdemucs")
    stems: int = Field(default=4, ge=2, le=6)
    device: str = Field(default="cuda")


class AnnotationConfig(BaseModel):
    """LLM annotation settings."""

    llm_provider: str = Field(default="anthropic")
    llm_model: str = Field(default="claude-sonnet-4-20250514")
    max_tokens: int = Field(default=2000, ge=100, le=16000)
    temperature: float = Field(default=0.3, ge=0.0, le=2.0)


class BenchmarkConfig(BaseModel):
    """Benchmark generation and validation settings."""

    val_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    test_ratio: float = Field(default=0.9, ge=0.0, le=1.0)
    tasks: list[str] = Field(
        default_factory=lambda: [
            "reasoning_qa",
            "source_emotion",
            "relationship",
            "symbolic_sound",
            "sound_world",
        ]
    )


class EvaluationConfig(BaseModel):
    """Model evaluation settings."""

    models: list[str] = Field(
        default_factory=lambda: [
            "claude-sonnet-4-20250514",
            "gpt-4o",
            "gemini-1.5-pro",
        ]
    )
    judge_model: str = Field(default="gpt-4o")


class ProjectConfig(BaseModel):
    """Top-level project settings."""

    name: str = Field(default="MusicWhy")
    seed: int = Field(default=42)
    data_root: str = Field(default="./data")
    num_workers: int = Field(default=8, ge=1)


class MusicWhyConfig(BaseModel):
    """Root configuration aggregating all sections."""

    project: ProjectConfig = Field(default_factory=ProjectConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    separation: SeparationConfig = Field(default_factory=SeparationConfig)
    annotation: AnnotationConfig = Field(default_factory=AnnotationConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge *override* into a copy of *base*.

    Lists in *override* replace the corresponding list in *base* entirely
    (no element-wise merge).  Scalar values are overwritten.  Dicts are
    merged recursively.
    """
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> MusicWhyConfig:
    """Load and validate a MusicWhy configuration.

    Parameters
    ----------
    path:
        Path to a YAML config file.  If *None*, only the built-in defaults
        are used.  If a path is given, its values are merged on top of the
        defaults from ``configs/default.yaml``.
    overrides:
        An optional dict of programmatic overrides applied last (handy for
        CLI ``--set key=value`` flags).

    Returns
    -------
    MusicWhyConfig
        A fully validated configuration object.
    """
    # Start with the built-in default config
    base_data: dict[str, Any] = {}
    if _DEFAULT_CONFIG.exists():
        with open(_DEFAULT_CONFIG, encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
            if isinstance(loaded, dict):
                base_data = loaded

    # Layer the user-supplied config on top
    if path is not None:
        user_path = Path(path)
        with open(user_path, encoding="utf-8") as fh:
            user_data = yaml.safe_load(fh)
        if isinstance(user_data, dict):
            base_data = merge_configs(base_data, user_data)

    # Apply programmatic overrides
    if overrides:
        base_data = merge_configs(base_data, overrides)

    return MusicWhyConfig.model_validate(base_data)
