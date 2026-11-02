from prompt_forensics.jsonparse import as_str_list, coerce_float, extract_json_object, strip_reasoning


def test_reasoning_blocks_are_removed():
    assert strip_reasoning('<think>{"a": 1}</think>{"b": 2}').strip() == '{"b": 2}'
    assert strip_reasoning("<think>never closed") == ""


def test_json_is_found_in_fences_and_prose():
    assert extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json_object('Sure! {"a": {"b": "}"}} trailing') == {"a": {"b": "}"}}
    assert extract_json_object("no json here") is None


def test_zero_is_kept():
    assert coerce_float(0) == 0.0
    assert coerce_float("0.0") == 0.0
    assert coerce_float(None) is None
    assert coerce_float("abc") is None


def test_string_lists():
    assert as_str_list("x") == ["x"]
    assert as_str_list([" a ", "", None, 3]) == ["a", "3"]
