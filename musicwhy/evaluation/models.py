"""Model wrappers for MusicWhy evaluation.

All wrappers implement a common interface (:class:`BaseModelWrapper`) so
experiments can iterate over models uniformly.  Wrappers handle loading,
audio pre-processing, and prompt formatting for each model family.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

logger = logging.getLogger(__name__)

AudioInput = Union[str, Path, np.ndarray, bytes]
"""Accepted audio input: file path, numpy waveform, or raw bytes."""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseModelWrapper(ABC):
    """Abstract interface that every evaluation model must implement."""

    name: str = "base"

    @abstractmethod
    def answer_mc(
        self,
        audio: AudioInput,
        question: str,
        options: Dict[str, str],
    ) -> str:
        """Answer a multiple-choice question about the given audio.

        Parameters
        ----------
        audio : AudioInput
            Audio file path, waveform array, or bytes.
        question : str
            The question text (includes feature descriptions).
        options : dict
            ``{"A": "...", "B": "...", ...}``

        Returns
        -------
        str
            The chosen option key (e.g., ``"A"``).
        """
        ...

    @abstractmethod
    def answer_open(
        self,
        audio: AudioInput,
        question: str,
    ) -> str:
        """Produce a free-form answer to an open-ended question.

        Returns
        -------
        str
            The model's response.
        """
        ...

    @abstractmethod
    def predict_source(
        self,
        audio: AudioInput,
        emotion_query: str,
    ) -> str:
        """Predict which source stem is most responsible for *emotion_query*.

        Returns
        -------
        str
            A stem name (e.g., ``"drums"``, ``"vocals"``).
        """
        ...

    @abstractmethod
    def explain_connection(
        self,
        audio: AudioInput,
        context: str,
        question: str,
    ) -> str:
        """Explain the connection between audio characteristics and context.

        Returns
        -------
        str
            A free-form explanation.
        """
        ...


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _load_audio_waveform(
    audio: AudioInput,
    target_sr: int = 16000,
) -> np.ndarray:
    """Load audio to a numpy waveform at *target_sr* Hz.

    Requires ``librosa`` for file loading and resampling.
    """
    if isinstance(audio, np.ndarray):
        return audio

    import librosa

    if isinstance(audio, (str, Path)):
        y, _ = librosa.load(str(audio), sr=target_sr, mono=True)
        return y

    if isinstance(audio, bytes):
        y, _ = librosa.load(io.BytesIO(audio), sr=target_sr, mono=True)
        return y

    raise TypeError(f"Unsupported audio type: {type(audio)}")


def _audio_to_base64(audio: AudioInput) -> str:
    """Encode audio as base64 for API calls."""
    if isinstance(audio, (str, Path)):
        with open(str(audio), "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    if isinstance(audio, bytes):
        return base64.b64encode(audio).decode("ascii")
    if isinstance(audio, np.ndarray):
        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, audio, samplerate=16000, format="WAV")
        return base64.b64encode(buf.getvalue()).decode("ascii")
    raise TypeError(f"Unsupported audio type: {type(audio)}")


def _format_mc_prompt(question: str, options: Dict[str, str]) -> str:
    """Format a multiple-choice question as a prompt string."""
    lines = [question, ""]
    for key in sorted(options.keys()):
        lines.append(f"{key}) {options[key]}")
    lines.append("")
    lines.append(
        "Respond with ONLY the letter of the correct answer (A, B, C, or D)."
    )
    return "\n".join(lines)


# Most specific first. Only capital letters count: the old parser took the first a/b/c/d anywhere,
# so "The answer is B" became "A" (from "answer") and "Based on..." became "B".
_MC_PATTERNS = [
    re.compile(r"^\W*([ABCD])\W*$"),                                      # "B", "(B)", "**B**", "B."
    re.compile(r"[Aa]nswer\s*(?:is|would be|:)?\s*[:\-]?\s*\**\(?([ABCD])\b"),  # "The answer is B", "Answer: (B)"
    re.compile(r"^\W*([ABCD])[).:]"),                                     # "B) horn sections", "B. ..."
    re.compile(r"\b(?:option|choice)\s+\(?([ABCD])\b", re.IGNORECASE),    # "option B"
    re.compile(r"\(([ABCD])\)"),                                          # "... (B) ..."
    re.compile(r"\b([ABCD])\)"),                                          # "... B) ..."
]
_last_mc = threading.local()


def last_mc_response() -> str:
    """Raw text behind this thread's most recent multiple-choice answer (for saving with results)."""
    return getattr(_last_mc, "text", "")


def _extract_mc_answer(response: str) -> str:
    """The option letter a response chose, or "X" when none can be found (scored as wrong, never guessed)."""
    _last_mc.text = response
    text = response.strip()
    for pattern in _MC_PATTERNS:
        m = pattern.search(text)
        if m:
            return m.group(1).upper()
    letters = set(re.findall(r"\b([ABCD])\b", text))
    return letters.pop() if len(letters) == 1 else "X"


# ---------------------------------------------------------------------------
# Qwen2-Audio-7B-Instruct
# ---------------------------------------------------------------------------

class Qwen2AudioWrapper(BaseModelWrapper):
    """Wrapper for Qwen2-Audio-7B-Instruct via HuggingFace Transformers.

    Requires ``transformers``, ``accelerate``, and ``torch``.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    device_map : str
        Torch device map strategy.
    max_new_tokens : int
        Maximum generation length.
    """

    name = "qwen2-audio"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-Audio-7B-Instruct",
        device_map: str = "auto",
        max_new_tokens: int = 1024,
    ) -> None:
        self.model_name = model_name
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self._model: Any = None
        self._processor: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

        logger.info("Loading %s ...", self.model_name)
        self._processor = AutoProcessor.from_pretrained(self.model_name)
        self._model = Qwen2AudioForConditionalGeneration.from_pretrained(
            self.model_name,
            device_map=self.device_map,
            dtype=torch.bfloat16,
        )
        logger.info("%s loaded.", self.model_name)

    def _load_clip(self, path: str, seconds: float = 30.0) -> Any:
        """Load a clip from the middle of the track; the encoder only sees 30 s anyway."""
        import librosa
        sr = self._processor.feature_extractor.sampling_rate
        duration = librosa.get_duration(path=path)
        offset = max(0.0, duration / 2 - seconds / 2)
        wave, _ = librosa.load(path, sr=sr, offset=offset, duration=seconds)
        return wave

    def _generate(self, messages: List[Dict[str, Any]], max_new_tokens: Optional[int] = None) -> str:
        import torch
        self._ensure_loaded()
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        clips = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                for part in msg["content"]:
                    if isinstance(part, dict) and part.get("type") == "audio":
                        clips.append(self._load_clip(str(part["audio_url"])))

        kwargs: Dict[str, Any] = {"text": text, "return_tensors": "pt", "padding": True}
        if clips:
            kwargs["audio"] = clips
            kwargs["sampling_rate"] = self._processor.feature_extractor.sampling_rate
        inputs = self._processor(**kwargs)
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self._model.generate(
                **inputs, max_new_tokens=max_new_tokens or self.max_new_tokens,
            )
        # Decode only newly generated tokens
        generated = output_ids[0][inputs["input_ids"].shape[-1]:]
        return self._processor.decode(generated, skip_special_tokens=True)

    def _build_messages(
        self, audio: AudioInput, text_prompt: str,
    ) -> List[Dict[str, Any]]:
        audio_path = str(audio) if isinstance(audio, (str, Path)) else ""
        content: List[Dict[str, Any]] = []
        if audio_path:
            content.append({"type": "audio", "audio_url": audio_path})
        content.append({"type": "text", "text": text_prompt})
        return [{"role": "user", "content": content}]

    # -- interface --

    def answer_mc(self, audio: AudioInput, question: str, options: Dict[str, str]) -> str:
        prompt = _format_mc_prompt(question, options)
        response = self._generate(self._build_messages(audio, prompt), max_new_tokens=32)
        return _extract_mc_answer(response)

    def answer_open(self, audio: AudioInput, question: str) -> str:
        return self._generate(self._build_messages(audio, question))

    def predict_source(self, audio: AudioInput, emotion_query: str) -> str:
        prompt = (
            f"Listen to this audio. Which instrument or stem (e.g., vocals, drums, "
            f"bass, guitar, piano, other) is MOST responsible for the feeling of "
            f"{emotion_query}? Respond with just the stem name."
        )
        return self._generate(self._build_messages(audio, prompt), max_new_tokens=16).strip().lower()

    def explain_connection(self, audio: AudioInput, context: str, question: str) -> str:
        prompt = (
            f"Context: {context}\n\nQuestion: {question}\n\n"
            f"Provide a detailed explanation connecting the audio characteristics "
            f"to the context."
        )
        return self._generate(self._build_messages(audio, prompt))


# ---------------------------------------------------------------------------
# Audio-Reasoner-7B
# ---------------------------------------------------------------------------

class AudioReasonerWrapper(Qwen2AudioWrapper):
    """Wrapper for Audio-Reasoner-7B (chain-of-thought variant of Qwen2-Audio).

    Uses the same interface as :class:`Qwen2AudioWrapper` with a different
    default model checkpoint.
    """

    name = "audio-reasoner"

    def __init__(
        self,
        model_name: str = "Sakshi20/Audio-Reasoner-7B",
        device_map: str = "auto",
        max_new_tokens: int = 1500,
    ) -> None:
        super().__init__(
            model_name=model_name,
            device_map=device_map,
            max_new_tokens=max_new_tokens,
        )


# ---------------------------------------------------------------------------
# Gemini (with audio support)
# ---------------------------------------------------------------------------

class GeminiAudioWrapper(BaseModelWrapper):
    """Wrapper for Google Gemini models with audio input.

    Requires the ``google-generativeai`` package and a valid
    ``GOOGLE_API_KEY`` environment variable.

    Parameters
    ----------
    model_name : str
        Gemini model identifier.
    """

    name = "gemini-audio"

    def __init__(self, model_name: str = "gemini-2.0-flash") -> None:
        self.model_name = model_name
        self._model: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import google.generativeai as genai

        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY environment variable is required.")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(self.model_name)
        logger.info("Gemini model %s initialized.", self.model_name)

    def _generate(self, audio: AudioInput, prompt: str) -> str:
        import google.generativeai as genai
        self._ensure_loaded()

        parts: list[Any] = []
        # Upload or inline the audio
        if isinstance(audio, (str, Path)) and Path(str(audio)).exists():
            audio_file = genai.upload_file(str(audio))
            parts.append(audio_file)
        elif isinstance(audio, bytes):
            parts.append({"mime_type": "audio/wav", "data": audio})
        elif isinstance(audio, np.ndarray):
            import soundfile as sf
            buf = io.BytesIO()
            sf.write(buf, audio, samplerate=16000, format="WAV")
            parts.append({"mime_type": "audio/wav", "data": buf.getvalue()})

        parts.append(prompt)
        response = self._model.generate_content(parts)
        return response.text

    def answer_mc(self, audio: AudioInput, question: str, options: Dict[str, str]) -> str:
        prompt = _format_mc_prompt(question, options)
        response = self._generate(audio, prompt)
        return _extract_mc_answer(response)

    def answer_open(self, audio: AudioInput, question: str) -> str:
        return self._generate(audio, question)

    def predict_source(self, audio: AudioInput, emotion_query: str) -> str:
        prompt = (
            f"Listen to this audio carefully. Which instrument or sound source "
            f"(vocals, drums, bass, guitar, piano, or other) is MOST responsible "
            f"for creating the feeling of {emotion_query}? "
            f"Respond with just the source name."
        )
        return self._generate(audio, prompt).strip().lower()

    def explain_connection(self, audio: AudioInput, context: str, question: str) -> str:
        prompt = (
            f"Context: {context}\n\nQuestion: {question}\n\n"
            f"Listen to the audio and explain the connection between its "
            f"characteristics and the given context in detail."
        )
        return self._generate(audio, prompt)


# ---------------------------------------------------------------------------
# GPT-4o (with audio support)
# ---------------------------------------------------------------------------

class GPT4oAudioWrapper(BaseModelWrapper):
    """Wrapper for OpenAI GPT-4o with audio input.

    Requires the ``openai`` package and a valid ``OPENAI_API_KEY``
    environment variable.

    Parameters
    ----------
    model_name : str
        OpenAI model identifier.
    """

    name = "gpt-4o-audio"

    def __init__(self, model_name: str = "gpt-4o-audio-preview") -> None:
        self.model_name = model_name
        self._client: Any = None

    def _ensure_loaded(self) -> None:
        if self._client is not None:
            return
        import openai

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY environment variable is required.")
        self._client = openai.OpenAI(api_key=api_key)
        logger.info("OpenAI client initialized for %s.", self.model_name)

    def _generate(self, audio: AudioInput, prompt: str) -> str:
        self._ensure_loaded()

        messages: List[Dict[str, Any]] = []
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]

        # Attach audio as base64
        if audio is not None:
            b64 = _audio_to_base64(audio)
            content.append({
                "type": "input_audio",
                "input_audio": {"data": b64, "format": "wav"},
            })

        messages.append({"role": "user", "content": content})

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0,
            max_tokens=1500,
        )
        return response.choices[0].message.content or ""

    def answer_mc(self, audio: AudioInput, question: str, options: Dict[str, str]) -> str:
        prompt = _format_mc_prompt(question, options)
        response = self._generate(audio, prompt)
        return _extract_mc_answer(response)

    def answer_open(self, audio: AudioInput, question: str) -> str:
        return self._generate(audio, question)

    def predict_source(self, audio: AudioInput, emotion_query: str) -> str:
        prompt = (
            f"Listen to this audio. Identify which instrument or source "
            f"(vocals, drums, bass, guitar, piano, other) is MOST responsible "
            f"for the feeling of {emotion_query}. "
            f"Respond with only the source name."
        )
        return self._generate(audio, prompt).strip().lower()

    def explain_connection(self, audio: AudioInput, context: str, question: str) -> str:
        prompt = (
            f"Context: {context}\n\nQuestion: {question}\n\n"
            f"Based on the audio provided, explain the connection between the "
            f"sonic characteristics and the given context."
        )
        return self._generate(audio, prompt)


# ---------------------------------------------------------------------------
# Claude (text-only with feature descriptions)
# ---------------------------------------------------------------------------

class ClaudeTextWrapper(BaseModelWrapper):
    """Wrapper for Anthropic Claude using text feature descriptions.

    Since Claude does not natively accept raw audio, the audio input is
    expected to be either a text description of the features or a path to
    a feature JSON file.

    Requires the ``anthropic`` package and a valid ``ANTHROPIC_API_KEY``
    environment variable.

    Parameters
    ----------
    model_name : str
        Anthropic model identifier.
    """

    name = "claude-text"

    def __init__(self, model_name: str = "claude-haiku-5-5") -> None:
        self.model_name = model_name
        self._client: Any = None
        self._last = threading.local()  # evaluation jobs call the API from several threads

    @property
    def last_stop_reason(self) -> Optional[str]:
        """stop_reason of this thread's most recent API response."""
        return getattr(self._last, "stop_reason", None)

    def _ensure_loaded(self) -> None:
        if self._client is not None:
            return
        import anthropic

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError("ANTHROPIC_API_KEY environment variable is required.")
        self._client = anthropic.Anthropic(api_key=api_key)
        logger.info("Anthropic client initialized for %s.", self.model_name)

    def _audio_to_text(self, audio: AudioInput) -> str:
        """Convert audio input to a text description of features.

        If *audio* is already a string that does not point to a file, it is
        treated as a pre-computed feature description.
        """
        if isinstance(audio, str):
            if not audio.strip():
                return ""
            path = Path(audio)
            if path.is_file():
                if path.suffix == ".json":
                    import json
                    with open(path, "r", encoding="utf-8") as f:
                        features = json.load(f)
                    return self._features_to_text(features)
                # Audio file: extract features
                try:
                    from musicwhy.benchmark.question_generator import _describe_features
                    import librosa
                    y, sr = librosa.load(str(path), sr=22050, mono=True)
                    features = self._quick_features(y, sr)
                    return _describe_features(features)
                except ImportError:
                    return f"[Audio file: {path.name}]"
            # Treat as pre-existing feature description text
            return audio

        if isinstance(audio, dict):
            return self._features_to_text(audio)

        return "[Audio input provided but cannot be converted to text for Claude]"

    @staticmethod
    def _features_to_text(features: Dict[str, Any]) -> str:
        """Convert a feature dict to natural language."""
        from musicwhy.benchmark.question_generator import _describe_features
        return _describe_features(features)

    @staticmethod
    def _quick_features(y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Extract a minimal feature set for text description."""
        import librosa
        from musicwhy.features.feature_to_language import dynamic_range_db
        tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(tempo_arr) if np.ndim(tempo_arr) == 0 else float(tempo_arr[0])
        sc = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        y_h, y_p = librosa.effects.hpss(y)
        hp = float(np.sum(y_h ** 2) / (np.sum(y_p ** 2) + 1e-10))
        rms = librosa.feature.rms(y=y)[0]
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_mean = np.mean(chroma, axis=1)

        return {
            "tempo": tempo,
            "spectral_centroid": sc.tolist(),
            "harmonic_percussive_ratio": hp,
            "dynamic_range": float(np.max(rms) - np.min(rms)),
            "dynamic_range_db": dynamic_range_db(rms),
            "estimated_key": int(np.argmax(chroma_mean)),
        }

    def _generate(self, text_prompt: str) -> str:
        self._ensure_loaded()
        # Thinking shares the max_tokens budget, so leave room for it and fail loudly on truncation
        response = self._client.messages.create(
            model=self.model_name,
            max_tokens=8000,
            messages=[{"role": "user", "content": text_prompt}],
        )
        text = "\n\n".join(b.text for b in response.content if b.type == "text")
        self._last.stop_reason = response.stop_reason
        if response.stop_reason == "max_tokens" or not text.strip():
            raise RuntimeError(f"incomplete response (stop_reason={response.stop_reason}, {len(text)} chars)")
        return text

    def answer_mc(self, audio: AudioInput, question: str, options: Dict[str, str]) -> str:
        feature_text = self._audio_to_text(audio)
        prompt = (
            f"Audio feature description:\n{feature_text}\n\n"
            + _format_mc_prompt(question, options)
        )
        response = self._generate(prompt)
        return _extract_mc_answer(response)

    def answer_open(self, audio: AudioInput, question: str) -> str:
        feature_text = self._audio_to_text(audio)
        prompt = f"Audio feature description:\n{feature_text}\n\n{question}"
        return self._generate(prompt)

    def predict_source(self, audio: AudioInput, emotion_query: str) -> str:
        feature_text = self._audio_to_text(audio)
        prompt = (
            f"Audio feature description:\n{feature_text}\n\n"
            f"Based on these audio features, which instrument or source "
            f"(vocals, drums, bass, guitar, piano, other) is MOST responsible "
            f"for the feeling of {emotion_query}? "
            f"Respond with only the source name."
        )
        return self._generate(prompt).strip().lower()

    def explain_connection(self, audio: AudioInput, context: str, question: str) -> str:
        feature_text = self._audio_to_text(audio)
        prompt = (
            f"Audio feature description:\n{feature_text}\n\n"
            f"Context: {context}\n\nQuestion: {question}\n\n"
            f"Explain the connection between the audio characteristics and "
            f"the given context."
        )
        return self._generate(prompt)


# ---------------------------------------------------------------------------
# MERT (music encoder -- embeddings only, for Exp 2)
# ---------------------------------------------------------------------------

class MERTWrapper(BaseModelWrapper):
    """Wrapper for MERT-v1-330M music encoder.

    MERT is an encoder-only model that produces embeddings, not text.
    The text-generation interface methods raise :class:`NotImplementedError`;
    use :meth:`get_embedding` directly for experiment 2.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    device : str
        Torch device (``"cuda"`` or ``"cpu"``).
    """

    name = "mert"

    def __init__(
        self,
        model_name: str = "m-a-p/MERT-v1-330M",
        device: str = "cuda",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: Any = None
        self._processor: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, Wav2Vec2FeatureExtractor

        logger.info("Loading MERT: %s ...", self.model_name)
        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.model_name, trust_remote_code=True,
        )
        self._model = AutoModel.from_pretrained(
            self.model_name, trust_remote_code=True,
        ).to(self.device)
        self._model.eval()
        # MERT's remote code returns hidden_states=None with current transformers (with the call
        # argument or the config flag), so capture each transformer layer's output with hooks
        self._layer_outputs: list = []
        for layer in self._model.encoder.layers:
            layer.register_forward_hook(
                lambda mod, inp, out: self._layer_outputs.append(out[0] if isinstance(out, tuple) else out)
            )
        logger.info("MERT loaded on %s.", self.device)

    def get_embedding(self, audio: AudioInput) -> np.ndarray:
        """Extract a fixed-size embedding from the audio.

        Parameters
        ----------
        audio : AudioInput
            Audio file path, waveform array, or bytes.

        Returns
        -------
        np.ndarray
            1-D embedding vector (mean-pooled over time frames from the
            last hidden state).
        """
        import torch
        self._ensure_loaded()

        sr = self._processor.sampling_rate
        waveform = _load_audio_waveform(audio, target_sr=sr)

        # 30 s chunks: attention memory grows with the square of length, so a whole 10-minute
        # song in one pass can exhaust GPU memory. Each chunk is mean-pooled over time in every
        # transformer layer; chunks are averaged weighted by length. Layers are then L2-normalised
        # and averaged, so each layer counts equally (deeper layers have larger magnitudes, and the
        # last layer alone is tuned to MERT's pretraining targets rather than general music traits).
        chunk = 30 * sr
        total, weight = None, 0
        for start in range(0, max(len(waveform), 1), chunk):
            piece = waveform[start:start + chunk]
            if len(piece) < sr and total is not None:
                break  # drop a trailing fragment under 1 s
            inputs = self._processor(piece, sampling_rate=sr, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            self._layer_outputs.clear()
            with torch.inference_mode():
                self._model(**inputs)
            layers = torch.stack(self._layer_outputs)  # (L, 1, T, D)
            pooled = layers.mean(dim=2).squeeze(1).float().cpu().numpy()  # (L, D)
            total = pooled * len(piece) if total is None else total + pooled * len(piece)
            weight += len(piece)
        per_layer = total / weight
        per_layer /= np.linalg.norm(per_layer, axis=1, keepdims=True) + 1e-8
        return per_layer.mean(axis=0)

    # -- text interface (not applicable for encoder-only model) --

    def answer_mc(self, audio: AudioInput, question: str, options: Dict[str, str]) -> str:
        raise NotImplementedError(
            "MERT is an encoder-only model and does not generate text. "
            "Use get_embedding() for experiment 2."
        )

    def answer_open(self, audio: AudioInput, question: str) -> str:
        raise NotImplementedError(
            "MERT is an encoder-only model. Use get_embedding()."
        )

    def predict_source(self, audio: AudioInput, emotion_query: str) -> str:
        raise NotImplementedError(
            "MERT is an encoder-only model. Use get_embedding()."
        )

    def explain_connection(self, audio: AudioInput, context: str, question: str) -> str:
        raise NotImplementedError(
            "MERT is an encoder-only model. Use get_embedding()."
        )
