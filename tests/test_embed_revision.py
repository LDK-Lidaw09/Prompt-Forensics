"""The embedding model is loaded at the revision used for the thesis."""

import sys
import types

from prompt_forensics.config import get_settings
from prompt_forensics.metrics import similarity

THESIS_REVISION = "104333d6af6f97649377c2afbde10a7704870c7b"


def _fake_sentence_transformers(monkeypatch, calls):
    module = types.ModuleType("sentence_transformers")

    def SentenceTransformer(name, **kwargs):
        calls.append((name, kwargs))
        return object()

    module.SentenceTransformer = SentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)


def test_default_revision_is_the_thesis_one(monkeypatch):
    monkeypatch.delenv("EMBED_MODEL_REVISION", raising=False)
    get_settings.cache_clear()
    assert get_settings().embed_model_revision == THESIS_REVISION
    get_settings.cache_clear()


def test_embedder_pins_revision_for_thesis_model_only(monkeypatch):
    calls = []
    _fake_sentence_transformers(monkeypatch, calls)
    similarity.get_embedder.cache_clear()
    similarity.get_embedder()
    similarity.get_embedder("some/other-model")
    similarity.get_embedder.cache_clear()
    assert calls[0][0] == "Alibaba-NLP/gte-large-en-v1.5"
    assert calls[0][1]["revision"] == get_settings().embed_model_revision
    assert calls[1][1]["revision"] is None
