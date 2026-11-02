from prompt_forensics.canonical import (
    build_interaction_payload,
    canonical_json,
    content_hash,
    normalize_text,
    prompt_id,
)


def payload(**overrides):
    base = dict(prompt="Hello  world", model="m", params={"temperature": 0.2, "seed": 1},
                timestamp="2026-01-01T00:00:00Z", user_id="u", session_id=None)
    base.update(overrides)
    return build_interaction_payload(**base)


def test_whitespace_is_collapsed_outside_quotes_only():
    assert normalize_text('a   b  "x    y"   c') == 'a b "x    y" c'


def test_quote_parity_survives_several_quoted_spans():
    text = 'say "a  b" then  "c   d"  end'
    assert normalize_text(text) == 'say "a  b" then "c   d" end'


def test_line_endings_and_unicode_are_normalised():
    composed, decomposed = "café", "café"
    assert normalize_text(decomposed + "\r\nx") == composed + "\nx"


def test_none_values_are_dropped_and_keys_sorted():
    assert canonical_json({"b": 1, "a": None, "c": " x "}) == '{"b":1,"c":"x"}'


def test_prompt_id_is_stable_under_formatting_changes():
    assert prompt_id(payload(prompt="Hello world")) == prompt_id(payload(prompt="  Hello   world \r\n"))


def test_prompt_id_changes_with_timestamp_but_content_hash_does_not():
    a, b = payload(), payload(timestamp="2026-01-02T00:00:00Z", user_id="other")
    assert prompt_id(a) != prompt_id(b)
    assert content_hash(a) == content_hash(b)


def test_content_hash_changes_with_generation_parameters():
    assert content_hash(payload()) != content_hash(payload(params={"temperature": 0.7, "seed": 1}))
