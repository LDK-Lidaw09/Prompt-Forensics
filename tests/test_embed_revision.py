"""The embedding model is loaded at the revision used for the thesis."""

import sys
import types

import pytest

from prompt_forensics.config import get_settings
from prompt_forensics.metrics import similarity

THESIS_REVISION = "104333d6af6f97649377c2afbde10a7704870c7b"


@pytest.fixture
def fresh_settings(monkeypatch):
    monkeypatch.delenv("EMBED_MODEL_NAME", raising=False)
    monkeypatch.delenv("EMBED_MODEL_REVISION", raising=False)
    get_settings.cache_clear()
    similarity.get_embedder.cache_clear()
    yield monkeypatch
    get_settings.cache_clear()
    similarity.get_embedder.cache_clear()


def _fake_sentence_transformers(monkeypatch, calls):
    module = types.ModuleType("sentence_transformers")

    def SentenceTransformer(name, **kwargs):
        calls.append((name, kwargs))
        return object()

    module.SentenceTransformer = SentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_default_revision_is_the_thesis_one(fresh_settings):
    assert get_settings().embed_model_revision == THESIS_REVISION
    assert similarity.embed_revision() == THESIS_REVISION


def test_empty_revision_means_latest(fresh_settings):
    fresh_settings.setenv("EMBED_MODEL_REVISION", "  ")
    get_settings.cache_clear()
    assert get_settings().embed_model_revision == ""
    assert similarity.embed_revision() is None


def test_embedder_pins_revision_for_thesis_model_only(fresh_settings):
    calls = []
    _fake_sentence_transformers(fresh_settings, calls)
    similarity.get_embedder()
    similarity.get_embedder("some/other-model")
    assert calls[0] == ("Alibaba-NLP/gte-large-en-v1.5", calls[0][1])
    assert calls[0][1]["revision"] == THESIS_REVISION
    assert calls[1][1]["revision"] is None
