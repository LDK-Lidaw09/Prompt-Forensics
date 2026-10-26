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
