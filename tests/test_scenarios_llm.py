import pytest
import requests

from prompt_forensics import llm
from prompt_forensics.scenarios import build_victim_prompt, load_scenarios, select_scenarios


def test_dataset_sizes_match_the_thesis():
    scenarios = load_scenarios("data/scenarios")
    assert len(scenarios) == 24_000
    assert len(select_scenarios(scenarios, "attacks")) == 21_600
    assert len(select_scenarios(scenarios, "safe")) == 24_000
    assert len({s["id"] for s in scenarios}) == 24_000


def test_selection_is_deterministic_and_filters_families():
    first = select_scenarios(load_scenarios("data/scenarios"), "attacks", "injection", limit=5)
    again = select_scenarios(load_scenarios("data/scenarios"), "attacks", "injection", limit=5)
    assert [s["id"] for s in first] == [s["id"] for s in again]
    assert {s["attack_type"] for s in first} == {"prompt_injection"}


def test_victim_prompt_layout():
    scenario = {"system_prompt": "S", "dataset": "D", "attack_prompt": "A", "safe_prompt": "P"}
    assert build_victim_prompt(scenario, "attacks") == "S\n\nD\n\nA"
    assert build_victim_prompt(scenario, "safe") == "S\n\nD\n\nP"


@pytest.mark.parametrize("model,expected", [
    ("Qwen/Qwen3-14B", ("vllm", "Qwen/Qwen3-14B")),
    ("lmstudio:mistral", ("lmstudio", "mistral")),
    ("sim:x", ("sim", "x")),
    ("unknown:thing", ("vllm", "unknown:thing")),
])
def test_model_prefixes(model, expected):
    assert llm.split_model_id(model) == expected


class FakeResponse:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


def test_openai_compatible_request(monkeypatch):
    sent = {}

    def fake_post(url, json=None, headers=None, timeout=None):
        sent.update(url=url, body=json)
        return FakeResponse(200, {"model": "m", "choices": [{"message": {"content": "hi"},
                                                             "finish_reason": "stop"}]})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    response = llm.call_llm("p", "m", {"seed": 5, "temperature": 0, "repetition_penalty": 1.3,
                                       "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}})
    assert response.text == "hi" and response.finish_reason == "stop"
    assert sent["url"].endswith("/chat/completions")
    assert sent["body"]["seed"] == 5 and sent["body"]["repetition_penalty"] == 1.3
    assert sent["body"]["chat_template_kwargs"] == {"enable_thinking": False}


def test_server_errors_raise(monkeypatch):
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: FakeResponse(500, {"error": "boom"}))
    with pytest.raises(llm.LLMError):
        llm.call_llm("p", "m")


def test_simulator_needs_no_server():
    assert llm.call_llm("hello", "sim:x").text.startswith("[simulator]")
