"""Shared fixtures. No test needs a GPU, a model server or the network."""

from __future__ import annotations

import pytest

from prompt_forensics import config, pipeline
from prompt_forensics.metrics import similarity
from prompt_forensics.signing import generate_key
from prompt_forensics.storage import EvidenceStore


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Point every setting that touches disk at a temporary directory."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "evidence.sqlite"))
    monkeypatch.setenv("SIGNING_KEY_PATH", str(tmp_path / "keys" / "key.pem"))
    monkeypatch.setenv("LIVE_MODEL", "sim:victim")
    monkeypatch.setenv("JUDGE_MODEL", "sim:judge")
    monkeypatch.setenv("ADVERSARY_MODEL", "sim:adversary")
    config.get_settings.cache_clear()
    pipeline.get_store.cache_clear()
    yield
    config.get_settings.cache_clear()
    pipeline.get_store.cache_clear()


@pytest.fixture
def store(tmp_path) -> EvidenceStore:
    return EvidenceStore(tmp_path / "store.sqlite", generate_key())


@pytest.fixture
def no_ml(monkeypatch):
    """Replace the embedding and BERTScore backends by cheap fakes."""
    monkeypatch.setattr(similarity, "cosine_pair", lambda a, b: 1.0 if a == b else 0.5)
    monkeypatch.setattr(similarity, "bertscore", lambda c, r: [(0.9, 0.9, 0.9)] * len(c))
