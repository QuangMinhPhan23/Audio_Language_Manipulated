"""Tests for musicwhy.utils — shared utilities."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from musicwhy.utils import (
    Timer,
    ensure_dir,
    get_device,
    load_json,
    save_json,
    setup_logging,
)


def test_setup_logging_returns_logger():
    logger = setup_logging("WARNING")
    assert isinstance(logger, logging.Logger)


def test_ensure_dir_creates(tmp_path: Path):
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert result.exists()
    assert result.is_dir()


def test_ensure_dir_existing(tmp_path: Path):
    result = ensure_dir(tmp_path)
    assert result == tmp_path


def test_save_and_load_json(tmp_path: Path):
    data = {"key": [1, 2, 3], "nested": {"a": True}}
    path = tmp_path / "test.json"
    save_json(data, path)
    loaded = load_json(path)
    assert loaded == data


def test_save_json_creates_parent(tmp_path: Path):
    path = tmp_path / "sub" / "dir" / "file.json"
    save_json({"x": 1}, path)
    assert path.exists()
    assert load_json(path) == {"x": 1}


def test_timer_context_manager():
    with Timer() as t:
        total = sum(range(1000))
    assert t.elapsed >= 0


def test_get_device_returns_string():
    device = get_device()
    assert device in ("cuda", "mps", "cpu")
