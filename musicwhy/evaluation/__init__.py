"""MusicWhy Model Evaluation Module (Phase 5).

Provides model wrappers, LLM-as-judge evaluation, experiment runners,
and metric calculations for the MusicWhy benchmark.
"""

from musicwhy.evaluation.metrics import (
    mc_accuracy,
    krippendorff_alpha,
    pearson_with_lag,
    mean_judge_score,
    bootstrap_ci,
    performance_gap,
)

__all__ = [
    "BaseModelWrapper",
    "Qwen2AudioWrapper",
    "AudioReasonerWrapper",
    "GeminiAudioWrapper",
    "GPT4oAudioWrapper",
    "ClaudeTextWrapper",
    "MERTWrapper",
    "LLMJudge",
    "run_exp1_benchmark",
    "run_exp2_features_vs_embeddings",
    "run_exp3_source_emotion_ablation",
    "run_exp4_sound_world_correlation",
    "run_exp5_instrumental_vs_vocal",
    "mc_accuracy",
    "krippendorff_alpha",
    "pearson_with_lag",
    "mean_judge_score",
    "bootstrap_ci",
    "performance_gap",
]

_LAZY_IMPORTS = {
    "BaseModelWrapper": "musicwhy.evaluation.models",
    "Qwen2AudioWrapper": "musicwhy.evaluation.models",
    "AudioReasonerWrapper": "musicwhy.evaluation.models",
    "GeminiAudioWrapper": "musicwhy.evaluation.models",
    "GPT4oAudioWrapper": "musicwhy.evaluation.models",
    "ClaudeTextWrapper": "musicwhy.evaluation.models",
    "MERTWrapper": "musicwhy.evaluation.models",
    "LLMJudge": "musicwhy.evaluation.judge",
    "run_exp1_benchmark": "musicwhy.evaluation.experiments",
    "run_exp2_features_vs_embeddings": "musicwhy.evaluation.experiments",
    "run_exp3_source_emotion_ablation": "musicwhy.evaluation.experiments",
    "run_exp4_sound_world_correlation": "musicwhy.evaluation.experiments",
    "run_exp5_instrumental_vs_vocal": "musicwhy.evaluation.experiments",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        mod = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
