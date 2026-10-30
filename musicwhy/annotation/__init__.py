"""
musicwhy.annotation -- Phase 3: Reasoning Annotation Pipeline

Modules
-------
llm_reasoning
    ReasoningGenerator with three approaches:
      1. API-based reasoning from extracted features (Claude / GPT-4o / Gemini)
      2. Local audio-LLM reasoning via Qwen2-Audio
      3. Cross-validation between two generators
relationship
    SongRelationshipAnnotator -- FAISS-powered similarity search and
    LLM-generated relationship explanations between song pairs.
batch_annotate
    BatchAnnotator -- orchestrates the full annotation pipeline with
    checkpointing, cost tracking, and progress reporting.
"""

__all__ = [
    "ReasoningConfig",
    "ReasoningGenerator",
    "features_to_description",
    "stem_comparison_description",
    "SongRelationshipAnnotator",
    "BatchAnnotator",
]

_LAZY_IMPORTS = {
    "ReasoningConfig": "musicwhy.annotation.llm_reasoning",
    "ReasoningGenerator": "musicwhy.annotation.llm_reasoning",
    "features_to_description": "musicwhy.annotation.llm_reasoning",
    "stem_comparison_description": "musicwhy.annotation.llm_reasoning",
    "SongRelationshipAnnotator": "musicwhy.annotation.relationship",
    "BatchAnnotator": "musicwhy.annotation.batch_annotate",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        mod = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
