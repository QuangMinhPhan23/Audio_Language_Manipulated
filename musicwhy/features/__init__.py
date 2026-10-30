"""
MusicWhy feature extraction and source separation — Phase 2.

Modules
-------
extract
    ``FeatureExtractor`` — multi-resolution audio feature extraction with
    HDF5 persistence.
separate
    ``SourceSeparator`` — Demucs-based stem separation and ablation mixes.
feature_to_language
    Rule-based translation of numerical features into natural-language
    descriptions for LLM reasoning pipelines.
"""

from musicwhy.features.feature_to_language import (
    ablation_description,
    features_to_description,
    stem_comparison_description,
)

__all__ = [
    "FeatureExtractor",
    "SourceSeparator",
    "ablation_description",
    "features_to_description",
    "stem_comparison_description",
]


def __getattr__(name: str):
    if name == "FeatureExtractor":
        from musicwhy.features.extract import FeatureExtractor
        return FeatureExtractor
    if name == "SourceSeparator":
        from musicwhy.features.separate import SourceSeparator
        return SourceSeparator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
