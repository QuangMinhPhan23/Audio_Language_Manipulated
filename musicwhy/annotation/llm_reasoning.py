"""
musicwhy.annotation.llm_reasoning
=================================

ReasoningGenerator with three approaches for generating chain-of-thought
annotations that explain *why* music sounds the way it does and what that
means culturally.

Approach 1 -- API-based (Claude / GPT-4o / Gemini)
    Sends extracted audio features as structured text to a frontier LLM.
Approach 2 -- Local audio LLM (Qwen2-Audio-7B-Instruct)
    Feeds raw audio directly to a multimodal model running locally.
Approach 3 -- Cross-validation
    Compares outputs from two generators and keeps only agreed-upon claims.

Also provides the feature-to-language translation helpers used by the
annotation pipeline.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature-to-language translation (Phase 3A helpers)
# ---------------------------------------------------------------------------

KEY_NAMES: List[str] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
]


def features_to_description(features: Dict[str, Any]) -> str:
    """Convert raw numerical audio features into a readable text paragraph.

    The output is designed to be consumed by an LLM that will reason about
    the emotional and cultural significance of the sonic properties.

    Parameters
    ----------
    features : dict
        Feature dictionary as produced by the Phase 2 extraction pipeline.
        Expected keys: ``tempo``, ``spectral_centroid``, ``spectral_bandwidth``,
        ``harmonic_percussive_ratio``, ``dynamic_range``, ``estimated_key``,
        ``key_profile``, ``mfcc_mean``, ``duration``.

    Returns
    -------
    str
        Multi-sentence description of the track's sonic character.
    """
    desc: List[str] = []

    # -- Tempo --
    tempo = features.get("tempo", 0.0)
    if tempo < 70:
        desc.append(
            f"Very slow tempo ({tempo:.0f} BPM, slower than resting heart rate)"
        )
    elif tempo < 100:
        desc.append(f"Slow-moderate tempo ({tempo:.0f} BPM, walking pace)")
    elif tempo < 130:
        desc.append(
            f"Moderate tempo ({tempo:.0f} BPM, typical pop/dance range)"
        )
    elif tempo < 150:
        desc.append(f"Fast tempo ({tempo:.0f} BPM, energetic)")
    else:
        desc.append(f"Very fast tempo ({tempo:.0f} BPM, intense/frantic)")

    # -- Spectral centroid (brightness) --
    sc_raw = features.get("spectral_centroid", [])
    sc_mean = float(np.mean(sc_raw)) if len(sc_raw) else 0.0
    if sc_mean < 1500:
        desc.append(
            f"Dark/warm sound (avg spectral centroid {sc_mean:.0f} Hz)"
        )
    elif sc_mean < 3000:
        desc.append(
            f"Balanced brightness (avg spectral centroid {sc_mean:.0f} Hz)"
        )
    else:
        desc.append(
            f"Bright/sharp sound (avg spectral centroid {sc_mean:.0f} Hz)"
        )

    # -- Spectral bandwidth --
    bw_raw = features.get("spectral_bandwidth", [])
    bw_mean = float(np.mean(bw_raw)) if len(bw_raw) else 0.0
    if bw_mean > 0:
        desc.append(f"Spectral bandwidth average {bw_mean:.0f} Hz")

    # -- Harmonic vs percussive balance --
    hp = features.get("harmonic_percussive_ratio", 1.0)
    if hp > 5:
        desc.append(
            "Dominated by sustained/tonal sounds (very harmonic)"
        )
    elif hp > 1:
        desc.append(
            "Mix of tonal and rhythmic elements (slightly harmonic)"
        )
    else:
        desc.append("Dominated by percussive/rhythmic elements")

    # -- Dynamic range --
    dr = features.get("dynamic_range_db")
    if dr is not None:
        if dr < 10.0:
            desc.append(
                "Compressed dynamics (consistent loudness)"
            )
        elif dr < 14.0:
            desc.append("Moderate dynamic range")
        else:
            desc.append(
                "Wide dynamic range (quiet and loud moments, expressive)"
            )

    # -- Key --
    key_idx = features.get("estimated_key", 0)
    if 0 <= key_idx < 12:
        desc.append(f"Estimated key center: {KEY_NAMES[key_idx]}")

    # -- Timbral summary from MFCCs --
    mfcc_mean = features.get("mfcc_mean", [])
    if len(mfcc_mean) >= 2:
        first_coeff = mfcc_mean[0]
        if first_coeff < -200:
            desc.append("Overall quiet / low-energy timbral signature")
        elif first_coeff > 0:
            desc.append("High-energy timbral signature")

    # -- Duration --
    dur = features.get("duration", 0.0)
    if dur > 0:
        desc.append(f"Duration: {dur / 60:.1f} minutes")

    return ". ".join(desc) + "."


def stem_comparison_description(
    full_features: Dict[str, Any],
    stem_features: Dict[str, Any],
    stem_name: str = "stem",
) -> str:
    """Describe what an individual stem contributes to the full mix.

    Compares the spectral, timbral, and energy properties of a separated
    stem against the corresponding full-mix features.

    Parameters
    ----------
    full_features : dict
        Features of the full mix.
    stem_features : dict
        Features of the isolated stem.
    stem_name : str
        Human label for the stem (e.g. ``"vocals"``, ``"drums"``).

    Returns
    -------
    str
        Multi-sentence comparison.
    """
    desc: List[str] = [f"[{stem_name.upper()}]"]

    full_sc = float(np.mean(full_features.get("spectral_centroid", [0])))
    stem_sc = float(np.mean(stem_features.get("spectral_centroid", [0])))
    if full_sc > 0:
        ratio = stem_sc / full_sc
        if ratio > 1.2:
            desc.append("Adds brightness to the mix")
        elif ratio < 0.8:
            desc.append("Adds warmth/depth to the mix")
        else:
            desc.append("Similar brightness to the overall mix")

    full_hp = full_features.get("harmonic_percussive_ratio", 1.0)
    stem_hp = stem_features.get("harmonic_percussive_ratio", 1.0)
    if stem_hp > full_hp * 1.5:
        desc.append("More harmonically dominant than the overall mix")
    elif stem_hp < full_hp * 0.5:
        desc.append("More percussive than the overall mix")

    stem_dr = stem_features.get("dynamic_range_db")
    full_dr = full_features.get("dynamic_range_db")
    if stem_dr is not None and full_dr is not None:
        if stem_dr > full_dr + 3:
            desc.append("Has wider dynamics than the full mix (expressive)")
        elif stem_dr < full_dr - 3:
            desc.append("Steadier dynamics than the full mix")

    stem_tempo = stem_features.get("tempo", 0.0)
    full_tempo = full_features.get("tempo", 0.0)
    if full_tempo > 0 and abs(stem_tempo - full_tempo) > 10:
        desc.append(
            f"Detected tempo ({stem_tempo:.0f} BPM) differs from mix "
            f"({full_tempo:.0f} BPM), suggesting rhythmic complexity"
        )

    return ". ".join(desc) + "."


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ReasoningConfig:
    """Configuration for :class:`ReasoningGenerator`.

    Parameters
    ----------
    provider : str
        LLM backend: ``"anthropic"``, ``"openai"``, or ``"google"``.
    model_name : str
        Model identifier, e.g. ``"claude-sonnet-4-20250514"``,
        ``"gpt-4o"``, ``"gemini-1.5-pro"``.
    max_tokens : int
        Maximum tokens for the LLM response.
    temperature : float
        Sampling temperature (0 = deterministic).
    batch_size : int
        How many tracks to process before yielding to the event loop
        for progress reporting.
    budget_limit_usd : float
        Hard cap on cumulative API spend.  The generator will refuse to
        make new calls once this is exceeded.
    max_retries : int
        Number of retries on transient API errors.
    base_retry_delay : float
        Base delay in seconds for exponential backoff.
    requests_per_minute : int
        Soft rate-limit (requests per minute) to stay within provider
        quotas.
    local_model_name : str
        HuggingFace model id for Approach 2 (local audio LLM).
    local_device_map : str
        ``device_map`` for ``from_pretrained`` (default ``"auto"``).
    """

    provider: Literal["anthropic", "openai", "google"] = "anthropic"
    model_name: str = "claude-sonnet-4-20250514"
    max_tokens: int = 8000
    temperature: float = 0.3
    batch_size: int = 10
    budget_limit_usd: float = 500.0
    max_retries: int = 5
    base_retry_delay: float = 1.0
    requests_per_minute: int = 50
    local_model_name: str = "Qwen/Qwen2-Audio-7B-Instruct"
    local_device_map: str = "auto"


# ---------------------------------------------------------------------------
# Cost estimation helpers
# ---------------------------------------------------------------------------

# Approximate $/1K tokens (input, output) as of late 2026.
# Update these as pricing changes.
_COST_PER_1K: Dict[str, Tuple[float, float]] = {
    "claude-sonnet-4-20250514": (0.003, 0.015),
    "claude-opus-4-20250514": (0.015, 0.075),
    "claude-haiku-5-5": (0.0008, 0.004),
    "claude-sonnet-5-5": (0.003, 0.015),
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gemini-1.5-pro": (0.00125, 0.005),
    "gemini-1.5-flash": (0.000075, 0.0003),
}


def _estimate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Return estimated cost in USD for a single LLM call."""
    rates = _COST_PER_1K.get(model, (0.003, 0.015))
    return (input_tokens / 1000) * rates[0] + (output_tokens / 1000) * rates[1]


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

REASONING_PROMPT_TEMPLATE = """\
You are a musicologist and cultural analyst. Given the following \
information about a song, provide detailed chain-of-thought reasoning.

SONG METADATA:
{song_metadata}

AUDIO ANALYSIS:
{feature_description}

INDIVIDUAL INSTRUMENT ANALYSIS:
{stem_descriptions}

WORLD CONTEXT (what was happening when this song was made/popular):
{world_context}

Generate the following analyses:

1. EMOTIONAL IMPACT ANALYSIS: Why does this song produce its emotional effect? \
Trace the feeling to specific audio features step by step.

2. SYMBOLIC SOUND IDENTIFICATION: What cultural or psychological associations \
do the specific sonic elements carry? Why?

3. SOUND-WORLD CONNECTION: How do this song's sonic characteristics reflect \
the cultural moment it emerged from?

Be specific. Reference the actual audio features provided. Don't be generic.
The world context is for the United States in the song's release year; the artist's own \
country may differ, so say when a link depends on that assumption. If the world context is \
empty, base section 3 on the genre and audio alone and say that no dated context was available. \
If a stem is described as silent, don't attribute any effect to it.\
"""

AUDIO_LLM_PROMPT_TEMPLATE = """\
Analyze this audio in detail.

Context: {metadata}
World context when this was made: {world_context}

1. What specific sounds, instruments, and production techniques do you hear?
2. Why might this audio produce the emotional effect it does?
3. What cultural associations do the sonic elements carry?
4. How does the sound connect to the era it was made in?

Be specific about what you hear. Reference exact moments if possible.\
"""

CROSS_VALIDATION_PROMPT = """\
You are an expert music analyst performing quality assurance. Two independent \
analyses of the same song are shown below. Your job is to compare them and \
extract only the claims that BOTH analyses support (with possibly different \
wording).

ANALYSIS A:
{reasoning_a}

ANALYSIS B:
{reasoning_b}

Return a JSON object with exactly these keys:
- "agreed_claims": list of strings, each a specific claim both analyses support
- "disputed_claims": list of strings, each a claim only ONE analysis makes
- "confidence": float 0-1, your overall confidence in the agreed claims
- "summary": string, a brief merged summary of the agreed-upon reasoning

Return ONLY valid JSON, no markdown fences.\
"""


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Simple token-bucket rate limiter for API calls."""

    def __init__(self, requests_per_minute: int) -> None:
        self._interval = 60.0 / max(requests_per_minute, 1)
        self._last_call: float = 0.0

    async def wait(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self._interval:
            await asyncio.sleep(self._interval - elapsed)
        self._last_call = time.monotonic()

    def wait_sync(self) -> None:
        now = time.monotonic()
        elapsed = now - self._last_call
        if elapsed < self._interval:
            time.sleep(self._interval - elapsed)
        self._last_call = time.monotonic()


# ---------------------------------------------------------------------------
# ReasoningGenerator
# ---------------------------------------------------------------------------

class ReasoningGenerator:
    """Generate chain-of-thought reasoning annotations for music tracks.

    Supports three complementary approaches that can be mixed and matched
    within the batch annotation pipeline.

    Parameters
    ----------
    config : ReasoningConfig
        All tuneable knobs (provider, model, budget, rate limits, ...).
    """

    def __init__(self, config: Optional[ReasoningConfig] = None) -> None:
        self.config = config or ReasoningConfig()
        self._limiter = _RateLimiter(self.config.requests_per_minute)
        self._total_cost_usd: float = 0.0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0
        self._call_count: int = 0

        # Lazily initialised API clients / local models
        self._api_client: Any = None
        self._local_model: Any = None
        self._local_processor: Any = None

    # -- Cost tracking -----------------------------------------------------

    @property
    def total_cost_usd(self) -> float:
        """Cumulative estimated API spend in USD."""
        return self._total_cost_usd

    @property
    def call_count(self) -> int:
        return self._call_count

    def cost_summary(self) -> Dict[str, Any]:
        """Return a summary dict of usage so far."""
        return {
            "total_cost_usd": round(self._total_cost_usd, 4),
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
            "call_count": self._call_count,
            "budget_remaining_usd": round(
                self.config.budget_limit_usd - self._total_cost_usd, 4
            ),
        }

    def _check_budget(self) -> None:
        if self._total_cost_usd >= self.config.budget_limit_usd:
            raise RuntimeError(
                f"Budget limit reached: ${self._total_cost_usd:.2f} "
                f">= ${self.config.budget_limit_usd:.2f}"
            )

    # -- Lazy client initialisation ----------------------------------------

    def _get_api_client(self) -> Any:
        """Return (or create) the provider-specific API client."""
        if self._api_client is not None:
            return self._api_client

        provider = self.config.provider
        if provider == "anthropic":
            import anthropic
            self._api_client = anthropic.Anthropic()
        elif provider == "openai":
            import openai
            self._api_client = openai.OpenAI()
        elif provider == "google":
            import google.generativeai as genai
            self._api_client = genai
        else:
            raise ValueError(f"Unknown provider: {provider}")

        logger.info("Initialised %s API client", provider)
        return self._api_client

    def _load_local_model(self) -> None:
        """Load Qwen2-Audio model and processor into GPU memory."""
        if self._local_model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        logger.info(
            "Loading local model %s (this may take a few minutes)...",
            self.config.local_model_name,
        )
        self._local_processor = AutoProcessor.from_pretrained(
            self.config.local_model_name
        )
        self._local_model = AutoModelForCausalLM.from_pretrained(
            self.config.local_model_name,
            device_map=self.config.local_device_map,
            torch_dtype=torch.float16,
        )
        logger.info("Local model loaded successfully")

    # -- Approach 1: API-based reasoning from features ---------------------

    def _call_api_with_retry(self, prompt: str) -> Tuple[str, int, int]:
        """Call the configured LLM provider with exponential backoff.

        Returns
        -------
        tuple of (response_text, input_tokens, output_tokens)
        """
        client = self._get_api_client()
        provider = self.config.provider
        last_exc: Optional[Exception] = None

        for attempt in range(1, self.config.max_retries + 1):
            try:
                self._limiter.wait_sync()

                if provider == "anthropic":
                    resp = client.messages.create(
                        model=self.config.model_name,
                        max_tokens=self.config.max_tokens,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    # A response can hold several text blocks (e.g. around thinking); keep all of them
                    text = "\n\n".join(b.text for b in resp.content if b.type == "text")
                    self.last_stop_reason = resp.stop_reason
                    if resp.stop_reason == "max_tokens":
                        logger.warning("Response truncated at max_tokens=%d", self.config.max_tokens)
                    in_tok = resp.usage.input_tokens
                    out_tok = resp.usage.output_tokens
                    return text, in_tok, out_tok

                elif provider == "openai":
                    resp = client.chat.completions.create(
                        model=self.config.model_name,
                        max_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    text = resp.choices[0].message.content
                    in_tok = resp.usage.prompt_tokens
                    out_tok = resp.usage.completion_tokens
                    return text, in_tok, out_tok

                elif provider == "google":
                    model = client.GenerativeModel(self.config.model_name)
                    generation_config = client.types.GenerationConfig(
                        max_output_tokens=self.config.max_tokens,
                        temperature=self.config.temperature,
                    )
                    resp = model.generate_content(
                        prompt,
                        generation_config=generation_config,
                    )
                    text = resp.text
                    # Gemini usage metadata
                    usage = getattr(resp, "usage_metadata", None)
                    in_tok = (
                        getattr(usage, "prompt_token_count", 0)
                        if usage
                        else 0
                    )
                    out_tok = (
                        getattr(usage, "candidates_token_count", 0)
                        if usage
                        else 0
                    )
                    return text, in_tok, out_tok

            except Exception as exc:
                last_exc = exc
                delay = self.config.base_retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "API call failed (attempt %d/%d): %s. "
                    "Retrying in %.1fs...",
                    attempt,
                    self.config.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

        raise RuntimeError(
            f"All {self.config.max_retries} API retries exhausted"
        ) from last_exc

    async def _call_api_with_retry_async(
        self, prompt: str
    ) -> Tuple[str, int, int]:
        """Async wrapper around the synchronous retry loop.

        Runs the blocking API call in an executor so it plays nicely with
        ``asyncio.gather`` in the batch pipeline.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None, self._call_api_with_retry, prompt
        )

    def generate_from_features(
        self,
        song_metadata: str,
        feature_description: str,
        stem_descriptions: str,
        world_context: str,
    ) -> str:
        """Approach 1 -- Generate reasoning from extracted audio features.

        Constructs the standard musicologist prompt and sends it to the
        configured API provider.

        Parameters
        ----------
        song_metadata : str
            Artist, title, year, genre, etc.
        feature_description : str
            Output of :func:`features_to_description`.
        stem_descriptions : str
            Concatenated :func:`stem_comparison_description` outputs.
        world_context : str
            Economic / cultural / political context for the track's era.

        Returns
        -------
        str
            Chain-of-thought reasoning text.

        Raises
        ------
        RuntimeError
            If the budget limit has been reached or all retries fail.
        """
        self._check_budget()

        prompt = REASONING_PROMPT_TEMPLATE.format(
            song_metadata=song_metadata,
            feature_description=feature_description,
            stem_descriptions=stem_descriptions,
            world_context=world_context,
        )

        text, in_tok, out_tok = self._call_api_with_retry(prompt)

        cost = _estimate_cost(self.config.model_name, in_tok, out_tok)
        self._total_cost_usd += cost
        self._total_input_tokens += in_tok
        self._total_output_tokens += out_tok
        self._call_count += 1

        logger.debug(
            "API call #%d: %d in / %d out tokens, est $%.4f",
            self._call_count,
            in_tok,
            out_tok,
            cost,
        )
        return text

    async def generate_from_features_async(
        self,
        song_metadata: str,
        feature_description: str,
        stem_descriptions: str,
        world_context: str,
    ) -> str:
        """Async variant of :meth:`generate_from_features`."""
        self._check_budget()

        prompt = REASONING_PROMPT_TEMPLATE.format(
            song_metadata=song_metadata,
            feature_description=feature_description,
            stem_descriptions=stem_descriptions,
            world_context=world_context,
        )

        text, in_tok, out_tok = await self._call_api_with_retry_async(prompt)

        cost = _estimate_cost(self.config.model_name, in_tok, out_tok)
        self._total_cost_usd += cost
        self._total_input_tokens += in_tok
        self._total_output_tokens += out_tok
        self._call_count += 1

        return text

    # -- Approach 2: Local audio LLM (Qwen2-Audio) ------------------------

    def generate_from_audio(
        self,
        audio_path: str,
        metadata: str,
        world_context: str,
    ) -> str:
        """Approach 2 -- Generate reasoning directly from audio.

        Loads Qwen2-Audio-7B-Instruct (on first call) and feeds it the
        raw audio file together with a structured text prompt.

        Parameters
        ----------
        audio_path : str
            Path to a WAV/MP3/FLAC file.
        metadata : str
            Artist, title, year, genre string.
        world_context : str
            Era context string.

        Returns
        -------
        str
            Chain-of-thought reasoning text.
        """
        import torch
        import librosa

        self._load_local_model()
        assert self._local_processor is not None
        assert self._local_model is not None

        prompt_text = AUDIO_LLM_PROMPT_TEMPLATE.format(
            metadata=metadata,
            world_context=world_context,
        )

        # Load audio at the sample rate the processor expects (16 kHz default)
        target_sr = getattr(
            self._local_processor, "sampling_rate", 16000
        )
        audio_array, _ = librosa.load(audio_path, sr=target_sr, mono=True)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_url": audio_path},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        text = self._local_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._local_processor(
            text=text,
            audios=[audio_array],
            return_tensors="pt",
            sampling_rate=target_sr,
        )
        inputs = {k: v.to(self._local_model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self._local_model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                do_sample=self.config.temperature > 0,
            )

        # Decode only the newly generated tokens
        input_len = inputs["input_ids"].shape[-1]
        generated = output_ids[0][input_len:]
        result = self._local_processor.decode(
            generated, skip_special_tokens=True
        )

        self._call_count += 1
        logger.debug(
            "Local model call #%d on %s (%d generated tokens)",
            self._call_count,
            audio_path,
            len(generated),
        )
        return result

    # -- Approach 3: Cross-validation --------------------------------------

    def cross_validate(
        self,
        reasoning_a: str,
        reasoning_b: str,
    ) -> Dict[str, Any]:
        """Approach 3 -- Cross-validate two independent reasoning outputs.

        Sends both analyses to the API provider and asks it to extract
        only the claims that both support.

        Parameters
        ----------
        reasoning_a : str
            First reasoning output (e.g. from Approach 1).
        reasoning_b : str
            Second reasoning output (e.g. from Approach 2 or a different
            provider).

        Returns
        -------
        dict
            ``{"agreed_claims": [...], "disputed_claims": [...],
            "confidence": float, "summary": str}``
        """
        self._check_budget()

        prompt = CROSS_VALIDATION_PROMPT.format(
            reasoning_a=reasoning_a,
            reasoning_b=reasoning_b,
        )

        text, in_tok, out_tok = self._call_api_with_retry(prompt)

        cost = _estimate_cost(self.config.model_name, in_tok, out_tok)
        self._total_cost_usd += cost
        self._total_input_tokens += in_tok
        self._total_output_tokens += out_tok
        self._call_count += 1

        # Parse the JSON response, with graceful fallback
        try:
            # Strip markdown fences if the model wrapped them anyway
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            result = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            logger.warning(
                "Cross-validation response was not valid JSON; "
                "returning raw text as summary."
            )
            result = {
                "agreed_claims": [],
                "disputed_claims": [],
                "confidence": 0.0,
                "summary": text,
            }

        # Ensure required keys exist
        for key in ("agreed_claims", "disputed_claims", "confidence", "summary"):
            result.setdefault(key, [] if "claims" in key else "")

        return result

    async def cross_validate_async(
        self,
        reasoning_a: str,
        reasoning_b: str,
    ) -> Dict[str, Any]:
        """Async variant of :meth:`cross_validate`."""
        self._check_budget()

        prompt = CROSS_VALIDATION_PROMPT.format(
            reasoning_a=reasoning_a,
            reasoning_b=reasoning_b,
        )

        text, in_tok, out_tok = await self._call_api_with_retry_async(prompt)

        cost = _estimate_cost(self.config.model_name, in_tok, out_tok)
        self._total_cost_usd += cost
        self._total_input_tokens += in_tok
        self._total_output_tokens += out_tok
        self._call_count += 1

        try:
            cleaned = text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
            result = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError):
            result = {
                "agreed_claims": [],
                "disputed_claims": [],
                "confidence": 0.0,
                "summary": text,
            }

        for key in ("agreed_claims", "disputed_claims", "confidence", "summary"):
            result.setdefault(key, [] if "claims" in key else "")
        return result

    # -- Cleanup -----------------------------------------------------------

    def unload_local_model(self) -> None:
        """Free GPU memory occupied by the local audio LLM."""
        if self._local_model is not None:
            import torch

            del self._local_model
            del self._local_processor
            self._local_model = None
            self._local_processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Local model unloaded and GPU cache cleared")
