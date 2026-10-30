"""Measure tempo, mode, lead instrument and vocals of a clip with tools independent of ACE-Step (§4).

Runs in the verify environment (jobs/phase0/install_verify.pbs). Each tool is loaded once, on first use.
Model files live in PHASE0_MODELS (default /srv/scratch/z5531827/.cache/phase0_models).
"""
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

MODEL_DIR = Path(os.environ.get("PHASE0_MODELS", "/srv/scratch/z5531827/.cache/phase0_models"))
EFFNET = "discogs-effnet-bs64-1.pb"
JAMENDO = "mtg_jamendo_instrument-discogs-effnet-1"
AUDIOSET_LABELS = "class_labels_indices.csv"
# Design instrument -> class name in each tagger's vocabulary
JAMENDO_NAMES = {"piano": "piano", "electric guitar": "electricguitar", "saxophone": "saxophone",
                 "synthesizer": "synthesizer", "violin": "violin", "trumpet": "trumpet"}
AUDIOSET_NAMES = {"piano": "Piano", "electric guitar": "Electric guitar", "saxophone": "Saxophone",
                  "synthesizer": "Synthesizer", "violin": "Violin, fiddle", "trumpet": "Trumpet"}
CLAP_MODEL = "clap_music"  # laion/larger_clap_music, saved to MODEL_DIR/clap_music/ by install_verify.pbs
CLAP_PROMPTS = {inst: f"music with {inst} as the lead instrument" for inst in JAMENDO_NAMES}
KEY_PROFILES = ("edma", "bgate")


def load_mono(path: Path, sr: int) -> np.ndarray:
    import librosa
    import soundfile as sf

    audio, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    return audio if file_sr == sr else librosa.resample(audio, orig_sr=file_sr, target_sr=sr, res_type="soxr_hq")


def clip_properties(audio: np.ndarray, sr: int, frame_s: float = 0.05, floor: float = 1e-3) -> Dict[str, float]:
    """Test 3.2 inputs: sample rate, duration and the longest run of near-silent frames (RMS below floor,
    about -60 dBFS). audio is [channels, samples] or [samples]."""
    mono = np.asarray(audio, dtype=np.float32)
    mono = mono.mean(axis=0) if mono.ndim == 2 else mono
    hop = max(int(frame_s * sr), 1)
    n = len(mono) // hop
    rms = np.sqrt(np.mean(mono[: n * hop].reshape(n, hop) ** 2, axis=1)) if n else np.zeros(0)
    quiet = rms < floor
    longest = run = 0
    for q in quiet:
        run = run + 1 if q else 0
        longest = max(longest, run)
    lead = int(np.argmin(quiet)) if not quiet.all() else len(quiet)  # quiet frames before the first sound
    trail = int(np.argmin(quiet[::-1])) if not quiet.all() else len(quiet)
    return {"sample_rate": sr, "duration": len(mono) / sr, "max_silence": longest * hop / sr,
            "leading_silence": lead * hop / sr, "trailing_silence": trail * hop / sr}


def bpm_from_beats(beats) -> Optional[float]:
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 4:
        return None
    return float(60.0 / np.median(np.diff(beats)))


class Measurer:
    def __init__(self, device: str = "cuda"):
        self.device = self._resolve_device(device)
        self._cache: Dict[str, Any] = {}
        self.tempo_a = self._first_tempo_tool()
        self.tempo_b = self._second_tempo_tool()

    @staticmethod
    def _resolve_device(device: str) -> str:
        """Fall back to CPU when the GPU's compute capability isn't supported by this torch build."""
        if device != "cuda":
            return device
        try:
            import torch
            if not torch.cuda.is_available():
                return "cpu"
            # Test a small convolution — the operation that fails on unsupported CC
            x = torch.randn(1, 1, 8, device="cuda")
            w = torch.randn(1, 1, 3, device="cuda")
            torch.nn.functional.conv1d(x, w)
            return "cuda"
        except Exception as e:
            print(f"GPU conv1d probe failed ({e}); running on CPU", flush=True)
            return "cpu"

    def _first_tempo_tool(self) -> str:
        """beat_this as planned. Its checkpoint is only published on a JKU server (no official mirror;
        third-party copies are unvetted pickles), which returned HTTP 503 on 2026-10-23. Without it,
        Essentia's RhythmExtractor2013 is the first tracker; rows record which tools measured them."""
        try:
            self._get("beat_this", self._load_beat_this)
            return "beat_this"
        except Exception as e:
            print(f"beat_this unavailable ({e}); using essentia_rhythm", flush=True)
            return "essentia_rhythm"

    def _load_beat_this(self):
        from beat_this.inference import File2Beats
        return File2Beats(checkpoint_path="final0", device=self.device, dbn=False)

    def _second_tempo_tool(self) -> str:
        """madmom's DBN tracker as planned. If madmom will not import (its last release predates Python 3.11
        and NumPy 2), a tracker different from the first: Essentia's, or librosa's when Essentia is first."""
        try:
            import madmom.features.beats  # noqa: F401
            return "madmom"
        except Exception:
            return "librosa" if self.tempo_a == "essentia_rhythm" else "essentia_rhythm"

    def _get(self, name, build):
        if name not in self._cache:
            self._cache[name] = build()
        return self._cache[name]

    # --- tempo ---
    def tempo(self, tool: str, path: Path) -> Optional[float]:
        if tool == "beat_this":
            beats, _ = self._get("beat_this", self._load_beat_this)(str(path))
            return bpm_from_beats(beats)
        if tool == "madmom":
            from madmom.audio.signal import Signal
            from madmom.features.beats import DBNBeatTrackingProcessor, RNNBeatProcessor
            rnn = self._get("madmom_rnn", RNNBeatProcessor)
            dbn = self._get("madmom_dbn", lambda: DBNBeatTrackingProcessor(fps=100))
            # Decoded samples, not the path: madmom cannot read ACE-Step's WAV files without ffmpeg
            return bpm_from_beats(dbn(rnn(Signal(load_mono(path, 44100), sample_rate=44100))))
        if tool == "essentia_rhythm":
            import essentia.standard as es
            rhythm = self._get("rhythm", lambda: es.RhythmExtractor2013(method="multifeature"))
            bpm = rhythm(load_mono(path, 44100))[0]
            return float(bpm) if bpm > 0 else None
        import librosa
        _, beats = librosa.beat.beat_track(y=load_mono(path, 22050), sr=22050, units="time")
        return bpm_from_beats(beats)

    # --- mode ---
    def modes(self, path: Path) -> Dict[str, str]:
        import essentia.standard as es
        audio = load_mono(path, 44100)
        out = {}
        for profile in KEY_PROFILES:
            extractor = self._get(f"key_{profile}", lambda p=profile: es.KeyExtractor(profileType=p))
            _, scale, _ = extractor(audio)
            out[profile] = scale
        return out

    # --- lead instrument ---
    def instruments_jamendo(self, path: Path) -> Dict[str, float]:
        import essentia.standard as es

        def build():
            classes = json.loads((MODEL_DIR / f"{JAMENDO}.json").read_text())["classes"]
            effnet = es.TensorflowPredictEffnetDiscogs(graphFilename=str(MODEL_DIR / EFFNET), output="PartitionedCall:1")
            head = es.TensorflowPredict2D(graphFilename=str(MODEL_DIR / f"{JAMENDO}.pb"), output="model/Sigmoid")
            return classes, effnet, head
        classes, effnet, head = self._get("jamendo", build)
        probs = np.asarray(head(effnet(load_mono(path, 16000)))).mean(axis=0)
        return {inst: float(probs[classes.index(name)]) for inst, name in JAMENDO_NAMES.items()}

    def instruments_passt(self, path: Path) -> Dict[str, float]:
        import torch

        def build():
            from hear21passt.base import get_basic_model
            model = get_basic_model(mode="logits").to(self.device).eval()
            with open(MODEL_DIR / AUDIOSET_LABELS, newline="") as f:
                index = {row["display_name"]: int(row["index"]) for row in csv.DictReader(f)}
            return model, {inst: index[name] for inst, name in AUDIOSET_NAMES.items()}
        model, idx = self._get("passt", build)
        audio = load_mono(path, 32000)
        win = 10 * 32000  # PaSST was trained on 10-second AudioSet clips
        chunks = [audio[i:i + win] for i in range(0, max(len(audio) - win, 0) + 1, win)]
        batch = torch.tensor(np.stack([np.pad(c, (0, win - len(c))) for c in chunks]), device=self.device)
        with torch.no_grad():
            probs = torch.sigmoid(model(batch)).mean(dim=0).cpu().numpy()
        return {inst: float(probs[i]) for inst, i in idx.items()}

    def instruments_clap(self, path: Path) -> Dict[str, float]:
        """Zero-shot CLAP: similarity between the clip and a text prompt per instrument, via the
        model's forward pass (which projects both modalities to the shared space)."""
        import torch

        def build():
            from transformers import ClapModel, ClapProcessor
            model_path = str(MODEL_DIR / CLAP_MODEL)
            model = ClapModel.from_pretrained(model_path).to(self.device).eval()
            processor = ClapProcessor.from_pretrained(model_path)
            instruments = list(CLAP_PROMPTS.keys())
            prompts = [CLAP_PROMPTS[i] for i in instruments]
            text_inputs = processor(text=prompts, return_tensors="pt", padding=True)
            text_kw = {k: v.to(self.device) for k, v in text_inputs.items()
                       if k in ("input_ids", "attention_mask")}
            return model, processor, text_kw, instruments

        model, processor, text_kw, instruments = self._get("clap", build)
        audio = load_mono(path, 48000)
        audio_inputs = processor(audio=[audio], sampling_rate=48000, return_tensors="pt", padding=True)
        audio_kw = {k: v.to(self.device) for k, v in audio_inputs.items()
                    if k in ("input_features", "is_longer")}
        with torch.no_grad():
            out = model(**text_kw, **audio_kw)
            sims = out.logits_per_audio.squeeze(0).cpu().numpy()  # [num_instruments]
        # Softmax over similarities → probabilities summing to 1
        exp_s = np.exp(sims - sims.max())
        probs = exp_s / exp_s.sum()
        return {inst: float(probs[i]) for i, inst in enumerate(instruments)}

    # --- vocals ---
    def vocal_ratio(self, path: Path) -> float:
        import soundfile as sf
        import torch
        from demucs.apply import apply_model

        def build():
            from demucs.pretrained import get_model
            return get_model("htdemucs").to(self.device).eval()
        model = self._get("demucs", build)
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        if sr != model.samplerate:
            import librosa
            audio = librosa.resample(audio.T, orig_sr=sr, target_sr=model.samplerate, res_type="soxr_hq").T
        if audio.shape[1] == 1:
            audio = np.repeat(audio, 2, axis=1)
        mix = torch.tensor(audio.T, device=self.device)
        with torch.no_grad():
            stems = apply_model(model, mix[None], device=self.device, split=True, overlap=0.25)[0]
        vocals = stems[model.sources.index("vocals")]
        rms = lambda x: float(torch.sqrt(torch.mean(x ** 2)))  # noqa: E731
        return rms(vocals) / max(rms(mix), 1e-9)

    def measure_clip(self, path: Path) -> Dict[str, Any]:
        path = Path(path)
        inst = {"jamendo": self.instruments_jamendo(path), "passt": self.instruments_passt(path)}
        try:
            inst["clap"] = self.instruments_clap(path)
        except Exception as e:
            print(f"  CLAP unavailable ({e}); proceeding with 2 taggers", flush=True)
        return {
            "tempo": {tool: self.tempo(tool, path) for tool in (self.tempo_a, self.tempo_b)},
            "mode": self.modes(path),
            "instrument": inst,
            "vocals": {"vocal_ratio": self.vocal_ratio(path)},
        }
