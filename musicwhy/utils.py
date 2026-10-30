"""Shared utilities for the MusicWhy project."""

from __future__ import annotations

import contextlib
import json
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure and return the ``musicwhy`` logger.

    Parameters
    ----------
    level:
        Logging level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).

    Returns
    -------
    logging.Logger
        The configured root logger for the ``musicwhy`` namespace.
    """
    from rich.logging import RichHandler

    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logger = logging.getLogger("musicwhy")
    if not logger.handlers:
        handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            markup=True,
        )
        handler.setLevel(numeric_level)
        fmt = logging.Formatter("%(message)s", datefmt="[%X]")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(numeric_level)
    return logger


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
def get_device() -> str:
    """Return the best available compute device as a PyTorch device string.

    Checks for CUDA and Apple MPS availability, falling back to ``"cpu"``.
    If PyTorch is not installed, ``"cpu"`` is returned.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    """Create *path* (and parents) if it does not exist, then return it.

    Parameters
    ----------
    path:
        Directory path to ensure exists.

    Returns
    -------
    Path
        The resolved directory path.
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p.resolve()


# ---------------------------------------------------------------------------
# Audio I/O wrapper
# ---------------------------------------------------------------------------
def load_audio(path: str | Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load an audio file with librosa and return ``(waveform, sample_rate)``.

    Parameters
    ----------
    path:
        Path to the audio file.
    sr:
        Target sample rate.  Pass ``None`` to preserve the native rate.

    Returns
    -------
    tuple[np.ndarray, int]
        A 1-D float32 waveform array and the actual sample rate.
    """
    import librosa

    y, sr_out = librosa.load(str(path), sr=sr, mono=True)
    return y, sr_out


# ---------------------------------------------------------------------------
# JSON I/O
# ---------------------------------------------------------------------------
def save_json(data: Any, path: str | Path, indent: int = 2) -> Path:
    """Serialize *data* as JSON to *path*.

    Parameters
    ----------
    data:
        JSON-serializable object.
    path:
        Destination file path.
    indent:
        JSON indentation level.

    Returns
    -------
    Path
        The written file path.
    """
    p = Path(path)
    ensure_dir(p.parent)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, ensure_ascii=False, default=str)
    return p


def load_json(path: str | Path) -> Any:
    """Read and return the JSON object stored at *path*.

    Parameters
    ----------
    path:
        Source file path.

    Returns
    -------
    Any
        The deserialized JSON object.
    """
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Timer context manager
# ---------------------------------------------------------------------------
class Timer:
    """Simple wall-clock timer usable as a context manager.

    Example
    -------
    >>> with Timer() as t:
    ...     do_work()
    >>> print(f"Took {t.elapsed:.2f}s")
    """

    def __init__(self) -> None:
        self.start: float = 0.0
        self.end: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self.start = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start


# ---------------------------------------------------------------------------
# Parallel progress bars
# ---------------------------------------------------------------------------
@contextlib.contextmanager
def tqdm_joblib(tqdm_bar: Any):
    """Patch joblib to update a tqdm progress bar during parallel execution.

    Usage
    -----
    >>> from tqdm.auto import tqdm
    >>> from joblib import Parallel, delayed
    >>> items = list(range(100))
    >>> with tqdm_joblib(tqdm(total=len(items), desc="Processing")):
    ...     results = Parallel(n_jobs=4)(delayed(fn)(i) for i in items)

    Parameters
    ----------
    tqdm_bar:
        An active ``tqdm`` progress bar instance.
    """
    import joblib

    class _TqdmCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            tqdm_bar.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = _TqdmCallback  # type: ignore[misc]
    try:
        yield tqdm_bar
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback  # type: ignore[misc]
        tqdm_bar.close()
