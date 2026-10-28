import sqlite3

import pytest

from prompt_forensics.canonical import build_interaction_payload
from prompt_forensics.signing import generate_key, load_or_create_key, public_key_path, verify_signature
from prompt_forensics.storage import EvidenceStore


def add(store, text="answer", prompt="question"):
    payload = build_interaction_payload(prompt=prompt, model="m", params={"seed": 1},
                                        timestamp=f"t-{text}")
    return store.append_interaction(payload=payload, response_text=text, metadata={"k": text})


def test_signature_round_trip():
    key = generate_key()
    digest = "ab" * 32
    signature = key.sign_digest(digest)
    assert verify_signature(key.public_key_pem, digest, signature)
    assert not verify_signature(key.public_key_pem, "cd" * 32, signature)
    assert not verify_signature(generate_key().public_key_pem, digest, signature)


def test_key_file_is_created_private_and_reused(tmp_path):
    path = tmp_path / "keys" / "k.pem"
    first = load_or_create_key(path)
    assert oct(path.stat().st_mode & 0o777) == "0o600"
    assert public_key_path(path).read_text() == first.public_key_pem
    assert load_or_create_key(path).public_key_pem == first.public_key_pem


def test_records_are_chained_and_verify(store):
    first, second = add(store, "one"), add(store, "two")
    row = store.get_interaction(second.prompt_id)
    assert row["prev_chain_hash"] == first.chain_hash
    report = store.verify()
    assert report.ok and report.checked == 2


def test_update_and_delete_are_refused(store):
    add(store)
    with sqlite3.connect(store.db_path) as conn:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute("UPDATE interactions SET response_text = 'x'")
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            conn.execute("DELETE FROM interactions")


def _tamper(store, sql):
    # An attacker with file access can drop the triggers; verification must still catch it.
    with sqlite3.connect(store.db_path) as conn:
        conn.execute("DROP TRIGGER interactions_no_update")
        conn.execute("DROP TRIGGER interactions_no_delete")
        conn.execute(sql)


def test_edited_response_is_detected(store):
    add(store, "one")
    _tamper(store, "UPDATE interactions SET response_text = 'forged' WHERE id = 1")
    assert any("response_hash" in p for p in store.verify().problems)


def test_deleted_row_breaks_the_chain(store):
    for text in ("one", "two", "three"):
        add(store, text)
    _tamper(store, "DELETE FROM interactions WHERE id = 2")
    assert any("chain broken" in p for p in store.verify().problems)


def test_rewritten_record_fails_against_trusted_key(tmp_path):
    trusted = generate_key()
    store = EvidenceStore(tmp_path / "s.sqlite", trusted)
    add(store)
    # Re-signing with another key passes the internal check but not the trusted key.
    forged = EvidenceStore(tmp_path / "s.sqlite", generate_key())
    add(forged, "forged")
    assert forged.verify().ok
    assert not forged.verify(trusted.public_key_pem).ok


def test_edited_raw_prompt_is_detected(store):
    add(store, "one")
    _tamper(store, "UPDATE interactions SET prompt_text = 'another question' WHERE id = 1")
    assert any("prompt_text" in p for p in store.verify().problems)


def test_edited_created_at_breaks_the_chain_hash(store):
    add(store, "one")
    _tamper(store, "UPDATE interactions SET created_at = '2020-01-01T00:00:00Z' WHERE id = 1")
    assert any("chain_hash does not match" in p for p in store.verify().problems)


def test_removed_last_records_need_an_anchor(store):
    for text in ("one", "two", "three"):
        add(store, text)
    anchor = store.anchor()
    _tamper(store, "DELETE FROM interactions WHERE id = 3")
    assert store.verify().ok  # the chain alone cannot see a cut at the end
    assert any("removed" in p for p in store.verify(anchors=[anchor]).problems)


def test_history_rewritten_with_the_key_is_caught_by_an_anchor(tmp_path):
    key = generate_key()
    path = tmp_path / "s.sqlite"
    store = EvidenceStore(path, key)
    for text in ("one", "two"):
        add(store, text)
    anchor = store.anchor()
    # The key holder rebuilds the whole database with a forged first answer.
    path.unlink()
    forged = EvidenceStore(path, key)
    for text in ("forged", "two"):
        add(forged, text)
    assert forged.verify(key.public_key_pem).ok
    assert not forged.verify(key.public_key_pem, anchors=[anchor]).ok
