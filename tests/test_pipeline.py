from prompt_forensics.pipeline import capture_interaction, get_store, replay_interaction


def test_capture_and_replay_with_the_simulator(no_ml):
    captured = capture_interaction("What is 2+2?", run_self_audit=True, session_id="s1")
    assert captured["signed"] and captured["response"].startswith("[simulator]")
    assert captured["audit"]["recommended_action"] == "manual_review"   # simulator is not JSON

    replayed = replay_interaction(captured["prompt_id"], seed=1234)
    assert replayed["baseline"] == captured["response"]
    assert replayed["content_hash"] == replayed["source_content_hash"]   # same input and parameters

    store = get_store()
    assert store.counts()["by_type"] == {"live": 1, "replay": 1}
    assert store.verify().ok

    # The second turn of the session sees the first one.
    second = capture_interaction("And times 3?", run_self_audit=False, session_id="s1")
    assert second["history_used"] and second["context_hash"]


def test_replay_reuses_the_recorded_configuration(monkeypatch):
    from prompt_forensics import config
    captured = capture_interaction("What is 2+2?", run_self_audit=False,
                                   temperature=0.7, max_tokens=64, seed=42)
    # The settings change after the capture: the replay must still use what was recorded.
    monkeypatch.setenv("LIVE_TEMPERATURE", "0.1")
    monkeypatch.setenv("LIVE_MAX_TOKENS", "999")
    config.get_settings.cache_clear()

    from prompt_forensics import pipeline
    sent = {}
    real_call = pipeline.call_llm

    def spy(prompt, model, params):
        sent.update(model=model, params=dict(params))
        return real_call(prompt, model, params)
    monkeypatch.setattr(pipeline, "call_llm", spy)

    replayed = replay_interaction(captured["prompt_id"])
    assert sent == {"model": "sim:victim", "params": {"temperature": 0.7, "max_tokens": 64, "seed": 42}}
    assert replayed["params"] == {"temperature": 0.7, "max_tokens": 64, "seed": 42}
    assert replayed["replay_config"]["faithful"] is True
    assert replayed["replay_config"]["divergences"] == []
    assert replayed["replay_config"]["not_recorded"]          # stated, never silently assumed
    assert replayed["content_hash"] == replayed["source_content_hash"]

    stored = get_store().get_interaction(replayed["prompt_id"])
    import json
    meta = json.loads(stored["metadata_json"])
    assert meta["source_prompt_id"] == captured["prompt_id"]
    assert meta["replay_config"]["recorded"]["params"] == {"temperature": 0.7, "max_tokens": 64, "seed": 42}


def test_replay_reports_every_divergence():
    captured = capture_interaction("What is 2+2?", run_self_audit=False,
                                   temperature=0.7, max_tokens=64, seed=42)
    replayed = replay_interaction(captured["prompt_id"], seed=43, model="sim:other")
    config = replayed["replay_config"]
    assert config["faithful"] is False
    fields = {d["field"]: d for d in config["divergences"]}
    assert fields["seed"] == {"field": "seed", "recorded": 42, "used": 43, "origin": "override"}
    assert fields["model"]["recorded"] == "sim:victim" and fields["model"]["origin"] == "override"
    assert fields["model_used"] == {"field": "model_used", "recorded": "victim", "used": "other", "origin": "served"}
    assert "temperature" not in fields and "max_tokens" not in fields
    assert replayed["content_hash"] != replayed["source_content_hash"]


def test_replay_of_a_record_without_parameters_says_so(store):
    from prompt_forensics.canonical import build_interaction_payload
    payload = build_interaction_payload(prompt="Hello", model="sim:victim", params={},
                                        timestamp="2026-01-01T00:00:00Z", user_id="u")
    stored = store.append_interaction(payload=payload, response_text="Hi", record_type="live")
    replayed = replay_interaction(stored.prompt_id, store=store)
    origins = {d["field"]: d["origin"] for d in replayed["replay_config"]["divergences"]}
    assert origins == {"temperature": "not_recorded", "max_tokens": "not_recorded", "seed": "not_recorded"}


def test_replay_refuses_a_payload_changed_after_capture(store):
    import sqlite3

    import pytest

    from prompt_forensics.canonical import build_interaction_payload
    payload = build_interaction_payload(prompt="Hello", model="sim:victim", params={"seed": 1},
                                        timestamp="2026-01-01T00:00:00Z", user_id="u")
    stored = store.append_interaction(payload=payload, response_text="Hi", record_type="live")
    with sqlite3.connect(store.db_path) as conn:
        for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type='trigger'").fetchall():
            conn.execute(f"DROP TRIGGER {name}")
        conn.execute("UPDATE interactions SET canonical_payload = replace(canonical_payload, '\"seed\":1', '\"seed\":2')")
    with pytest.raises(ValueError, match="does not match"):
        replay_interaction(stored.prompt_id, store=store)


def test_parameters_the_provider_ignores_are_reported_as_not_sent():
    from prompt_forensics.llm import sent_param_keys
    from prompt_forensics.pipeline import compare_configuration
    recorded = {"model": "anthropic:m", "model_used": None, "system": None,
                "params": {"temperature": 0.2, "max_tokens": 512, "seed": 7}}
    params = dict(recorded["params"])
    config = compare_configuration(recorded, "anthropic:m", params, {}, sent_keys=sent_param_keys("anthropic:m"))
    assert config["faithful"] is False
    assert config["divergences"] == [{"field": "seed", "recorded": 7, "used": None, "origin": "not_sent"}]


def test_replay_sends_the_recorded_system_message(store, monkeypatch):
    from prompt_forensics import pipeline
    from prompt_forensics.canonical import build_interaction_payload
    payload = build_interaction_payload(prompt="Hello", model="sim:victim", params={"seed": 1},
                                        timestamp="2026-01-01T00:00:00Z", user_id="u", system="Be brief.")
    stored = store.append_interaction(payload=payload, response_text="Hi", record_type="live")
    sent = {}
    real_call = pipeline.call_llm

    def spy(prompt, model, params):
        sent.update(params)
        return real_call(prompt, model, params)
    monkeypatch.setattr(pipeline, "call_llm", spy)
    replayed = replay_interaction(stored.prompt_id, store=store)
    assert sent["system"] == "Be brief."
    assert "system" not in {d["field"] for d in replayed["replay_config"]["divergences"]}
