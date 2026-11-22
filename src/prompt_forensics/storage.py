"""Append-only, hash-chained evidence store (thesis section 4.2.3).

All captured interactions live in one SQLite file, so that the whole evidence
base can be copied, hashed and handed over as a single artefact.

Integrity guarantees, from weakest to strongest:

1. **Append-only tables.** SQLite triggers reject every ``UPDATE`` and
   ``DELETE`` on evidence tables. A verdict computed later is added as a new
   row in ``annotations``; the original interaction is never rewritten.
2. **Self-describing hashes.** Each interaction keeps its canonical payload,
   so its Prompt ID, content hash and response hash can be recomputed.
3. **Hash chain.** Each row's ``chain_hash`` covers every stored field of
   the row (through their hashes) and the previous row's ``chain_hash``.
   Editing, deleting or reordering rows breaks the chain.
4. **Signatures.** Each ``chain_hash`` is signed with an ECDSA key kept outside
   the database (see :mod:`prompt_forensics.signing`). Someone who edits the
   file directly (bypassing the triggers) cannot produce valid signatures.
5. **Anchors.** The chain alone cannot show that the *last* rows were
   removed, and whoever holds the signing key can rebuild and re-sign the
   whole chain. :meth:`EvidenceStore.anchor` returns the head of the chain
   (row count and last ``chain_hash``); kept outside the system, for example
   with an RFC 3161 timestamp (``scripts/anchor_evidence.py``), it fixes the
   history up to that point.

Annotations (the verdicts and reports of the judges) have a chain of their
own, with the same layers: each one carries the hash of its result, the
parameters of the judge, the ``chain_hash`` and ``response_hash`` of the
interaction it judges, the ``chain_hash`` of the previous annotation and a
signature. Changing a verdict, a report or the answer that a verdict refers
to therefore breaks the annotation chain or its binding to the interaction.
The anchor also fixes the head of the annotation chain. Annotations written
before this chain existed (empty chain fields) are listed as unchained.

:meth:`EvidenceStore.verify` checks these properties and reports every
problem it finds.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Mapping

from .canonical import canonical_json, content_hash, normalize_text, prompt_id, sha256_hex, text_hash
from .signing import SigningKey, verify_signature

GENESIS_HASH = "0" * 64
APPEND_ONLY_TABLES = ("interactions", "annotations", "messages")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt_id         TEXT NOT NULL UNIQUE,
    content_hash      TEXT NOT NULL,
    created_at        TEXT NOT NULL,
    record_type       TEXT NOT NULL,      -- live | replay | attack
    user_id           TEXT,
    session_id        TEXT,
    model_requested   TEXT NOT NULL,
    model_used        TEXT,
    provider          TEXT,
    prompt_text       TEXT NOT NULL,
    response_text     TEXT NOT NULL,
    response_hash     TEXT NOT NULL,
    canonical_payload TEXT NOT NULL,      -- exact bytes hashed into prompt_id
    metadata_json     TEXT NOT NULL,      -- extra facts (timing, usage, attack info)
    prev_chain_hash   TEXT NOT NULL,
    chain_hash        TEXT NOT NULL,
    signature         TEXT,
    public_key_pem    TEXT
);
CREATE INDEX IF NOT EXISTS idx_interactions_content ON interactions(content_hash);

CREATE TABLE IF NOT EXISTS annotations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    interaction_id  INTEGER NOT NULL REFERENCES interactions(id),
    created_at      TEXT NOT NULL,
    kind            TEXT NOT NULL,        -- self_audit | investigator | ...
    model           TEXT,
    result_json     TEXT NOT NULL,
    result_hash     TEXT NOT NULL,
    params_json     TEXT,                 -- judge settings and prompt version
    target_chain_hash    TEXT,            -- chain_hash of the judged interaction
    target_response_hash TEXT,            -- response_hash of the judged interaction
    prev_chain_hash TEXT,
    chain_hash      TEXT,
    signature       TEXT,
    public_key_pem  TEXT
);
CREATE INDEX IF NOT EXISTS idx_annotations_interaction ON annotations(interaction_id);

CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,
    content     TEXT NOT NULL,
    canonical   TEXT,
    created_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
"""


def utc_now() -> str:
    """Current UTC time as an ISO 8601 string with microseconds."""
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _append_only_triggers() -> str:
    statements = []
    for table in APPEND_ONLY_TABLES:
        for action in ("UPDATE", "DELETE"):
            statements.append(
                f"CREATE TRIGGER IF NOT EXISTS {table}_no_{action.lower()} "
                f"BEFORE {action} ON {table} "
                f"BEGIN SELECT RAISE(ABORT, '{table} is append-only'); END;"
            )
    return "\n".join(statements)


# Columns of ``interactions`` covered by ``chain_hash`` as they are; the long
# texts are covered through their SHA-256 (see ``chain_digest``).
CHAINED_FIELDS = (
    "prompt_id", "content_hash", "created_at", "record_type", "user_id", "session_id",
    "model_requested", "model_used", "provider", "response_hash", "prev_chain_hash",
)


# Columns of ``annotations`` added after the first version of the store; older
# files get them (empty) when they are opened.
ANNOTATION_CHAIN_COLUMNS = ("params_json", "target_chain_hash", "target_response_hash",
                            "prev_chain_hash", "chain_hash", "signature", "public_key_pem")
ANNOTATION_CHAINED_FIELDS = (
    "interaction_id", "created_at", "kind", "model", "result_hash", "target_chain_hash",
    "target_response_hash", "prev_chain_hash",
)


def annotation_digest(row: Mapping[str, Any]) -> str:
    """Digest of one annotation: its fields, its result, its judge settings and its target."""
    record = {name: row[name] for name in ANNOTATION_CHAINED_FIELDS}
    record["params_sha256"] = sha256_hex(row["params_json"] or "")
    return sha256_hex(canonical_json(record))


def chain_digest(row: Mapping[str, Any]) -> str:
    """Digest that covers every stored field of a row and links it to the previous one.

    Only ``id`` (assigned by SQLite), ``chain_hash`` itself, ``signature`` and
    ``public_key_pem`` are left out. ``canonical_payload`` is covered by
    ``prompt_id``, ``response_text`` by ``response_hash``; ``prompt_text``
    (the prompt exactly as received, before normalisation) and
    ``metadata_json`` enter through their own SHA-256.
    """
    record = {name: row[name] for name in CHAINED_FIELDS}
    record["prompt_text_sha256"] = sha256_hex(row["prompt_text"])
    record["metadata_sha256"] = sha256_hex(row["metadata_json"])
    return sha256_hex(canonical_json(record))


@dataclass
class StoredInteraction:
    """What :meth:`EvidenceStore.append_interaction` returns."""

    id: int
    prompt_id: str
    content_hash: str
    response_hash: str
    chain_hash: str
    signed: bool


@dataclass
class VerificationReport:
    """Outcome of :meth:`EvidenceStore.verify`."""

    checked: int = 0
    problems: list[str] = field(default_factory=list)
    annotations_checked: int = 0
    unchained_annotations: int = 0

    @property
    def ok(self) -> bool:
        return not self.problems


class EvidenceStore:
    """Thin wrapper around the SQLite evidence file."""

    def __init__(self, db_path: str | Path, signing_key: SigningKey | None = None, read_only: bool = False):
        """Open (or create) the evidence file.

        With ``read_only=True`` the file is neither created nor changed, which
        is how a verifier should open an evidence file it was handed.
        """
        self.db_path = str(db_path)
        self.signing_key = signing_key
        self.read_only = read_only
        if read_only:
            return
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            present = {r["name"] for r in conn.execute("PRAGMA table_info(annotations)")}
            for column in ANNOTATION_CHAIN_COLUMNS:
                if column not in present:
                    conn.execute(f"ALTER TABLE annotations ADD COLUMN {column} TEXT")
            conn.executescript(_append_only_triggers())

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, timeout=30)
        else:
            conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ------------------------------------------------------------------ write

    def append_interaction(
        self,
        *,
        payload: Mapping[str, Any],
        response_text: str,
        record_type: str = "live",
        model_used: str | None = None,
        provider: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> StoredInteraction:
        """Store one interaction built with ``canonical.build_interaction_payload``."""
        canonical = canonical_json(payload)
        pid = prompt_id(payload)
        chash = content_hash(payload)
        rhash = text_hash(response_text)
        metadata_json = json.dumps(dict(metadata or {}), ensure_ascii=False, sort_keys=True)

        with self._connect() as conn:
            # BEGIN IMMEDIATE takes the write lock now, so that two writers
            # cannot both read the same "last chain hash" and fork the chain.
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT chain_hash FROM interactions ORDER BY id DESC LIMIT 1"
            ).fetchone()
            prev = row["chain_hash"] if row else GENESIS_HASH
            record = {
                "prompt_id": pid, "content_hash": chash, "created_at": utc_now(),
                "record_type": record_type, "user_id": payload.get("user_id"),
                "session_id": payload.get("session_id"), "model_requested": str(payload.get("model", "")),
                "model_used": model_used, "provider": provider,
                "prompt_text": str(payload.get("prompt", "")), "response_hash": rhash,
                "metadata_json": metadata_json, "prev_chain_hash": prev,
            }
            digest = chain_digest(record)
            signature = self.signing_key.sign_digest(digest) if self.signing_key else None
            public_key = self.signing_key.public_key_pem if self.signing_key else None
            cursor = conn.execute(
                """
                INSERT INTO interactions (
                    prompt_id, content_hash, created_at, record_type, user_id, session_id,
                    model_requested, model_used, provider, prompt_text, response_text,
                    response_hash, canonical_payload, metadata_json, prev_chain_hash,
                    chain_hash, signature, public_key_pem
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pid,
                    chash,
                    record["created_at"],
                    record_type,
                    record["user_id"],
                    record["session_id"],
                    record["model_requested"],
                    model_used,
                    provider,
                    record["prompt_text"],
                    response_text,
                    rhash,
                    canonical,
                    metadata_json,
                    prev,
                    digest,
                    signature,
                    public_key,
                ),
            )
            new_id = int(cursor.lastrowid)
        return StoredInteraction(new_id, pid, chash, rhash, digest, signature is not None)

    def add_annotation(
        self, interaction_id: int, kind: str, result: Mapping[str, Any], model: str | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> int:
        """Attach a verdict (self-audit, judge, report, ...) to a stored interaction.

        The annotation is chained, signed and bound to the interaction it
        judges (its ``chain_hash`` and ``response_hash``). ``params`` records
        the settings of the judge (temperature, seed, prompt version, ...).
        A repeated judgement is a new annotation; earlier ones are kept.
        """
        result_json = json.dumps(dict(result), ensure_ascii=False, sort_keys=True)
        params_json = canonical_json(dict(params)) if params else ""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            target = conn.execute(
                "SELECT chain_hash, response_hash FROM interactions WHERE id = ?", (interaction_id,)
            ).fetchone()
            if target is None:
                raise KeyError(f"no interaction {interaction_id}")
            last = conn.execute(
                "SELECT chain_hash FROM annotations WHERE chain_hash IS NOT NULL ORDER BY id DESC LIMIT 1"
            ).fetchone()
            record = {
                "interaction_id": interaction_id, "created_at": utc_now(), "kind": kind, "model": model,
                "result_hash": sha256_hex(result_json), "params_json": params_json,
                "target_chain_hash": target["chain_hash"], "target_response_hash": target["response_hash"],
                "prev_chain_hash": last["chain_hash"] if last else GENESIS_HASH,
            }
            digest = annotation_digest(record)
            signature = self.signing_key.sign_digest(digest) if self.signing_key else None
            public_key = self.signing_key.public_key_pem if self.signing_key else None
            cursor = conn.execute(
                "INSERT INTO annotations (interaction_id, created_at, kind, model, result_json, result_hash, "
                "params_json, target_chain_hash, target_response_hash, prev_chain_hash, chain_hash, "
                "signature, public_key_pem) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (interaction_id, record["created_at"], kind, model, result_json, record["result_hash"],
                 params_json, record["target_chain_hash"], record["target_response_hash"],
                 record["prev_chain_hash"], digest, signature, public_key),
            )
            return int(cursor.lastrowid)

    def append_message(
        self, session_id: str, role: str, content: str, canonical: str | None = None
    ) -> None:
        """Add one chat turn to a session transcript."""
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO messages (session_id, role, content, canonical, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, role, content, canonical or "", utc_now()),
            )

    # ------------------------------------------------------------------- read

    def recent_interactions(self, limit: int = 200) -> list[dict[str, Any]]:
        """Most recent interactions first, without their full payload."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, prompt_id, content_hash, created_at, record_type, model_requested, "
                "model_used, prompt_text FROM interactions ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_interaction(self, prompt_id_value: str) -> dict[str, Any] | None:
        """Full row for one Prompt ID, or None."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM interactions WHERE prompt_id = ?", (prompt_id_value,)
            ).fetchone()
        return dict(row) if row else None

    def annotations_for(self, interaction_id: int) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM annotations WHERE interaction_id = ? ORDER BY id",
                (interaction_id,),
            ).fetchall()
        return [dict(r) for r in rows]

    def session_messages(self, session_id: str, window: int) -> list[dict[str, Any]]:
        """Last ``window`` messages of a session, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content, canonical FROM messages WHERE session_id = ? "
                "ORDER BY id DESC LIMIT ?",
                (session_id, window),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def counts(self) -> dict[str, Any]:
        """Summary numbers for the dashboard."""
        with self._connect() as conn:
            total = conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            by_type = conn.execute(
                "SELECT record_type, COUNT(*) AS n FROM interactions GROUP BY record_type"
            ).fetchall()
            by_model = conn.execute(
                "SELECT COALESCE(model_used, model_requested) AS model, COUNT(*) AS n "
                "FROM interactions GROUP BY model ORDER BY n DESC"
            ).fetchall()
        return {
            "total": int(total),
            "by_type": {r["record_type"]: int(r["n"]) for r in by_type},
            "by_model": [(r["model"], int(r["n"])) for r in by_model],
        }

    # ----------------------------------------------------------------- verify

    def anchor(self) -> dict[str, Any]:
        """Head of the chain: keep it outside the system to fix the history so far."""
        with self._connect() as conn:
            rows = conn.execute("SELECT chain_hash FROM interactions ORDER BY id").fetchall()
            notes = conn.execute(
                "SELECT chain_hash FROM annotations WHERE chain_hash IS NOT NULL ORDER BY id").fetchall()
        return {
            "records": len(rows),
            "chain_hash": rows[-1]["chain_hash"] if rows else GENESIS_HASH,
            "annotations": len(notes),
            "annotation_chain_hash": notes[-1]["chain_hash"] if notes else GENESIS_HASH,
            "created_at": utc_now(),
        }

    def verify(self, trusted_public_key_pem: str | None = None,
               anchors: list[Mapping[str, Any]] | None = None) -> VerificationReport:
        """Recompute every hash, walk the chain and check every signature.

        If ``trusted_public_key_pem`` is given, signatures must verify against
        that key; otherwise the key stored in each row is used, which only
        proves internal consistency. Each anchor (from :meth:`anchor`, kept
        outside the database) must still be in the chain: the record at its
        position must exist and carry the anchored ``chain_hash``.
        """
        report = VerificationReport()
        expected_prev = GENESIS_HASH
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM interactions ORDER BY id").fetchall()

        for row in rows:
            report.checked += 1
            rid = f"interaction {row['id']}"
            payload = json.loads(row["canonical_payload"])

            if canonical_json(payload) != row["canonical_payload"]:
                report.problems.append(f"{rid}: canonical payload is not canonical")
            if prompt_id(payload) != row["prompt_id"]:
                report.problems.append(f"{rid}: prompt_id does not match the payload")
            if content_hash(payload) != row["content_hash"]:
                report.problems.append(f"{rid}: content_hash does not match the payload")
            if text_hash(row["response_text"]) != row["response_hash"]:
                report.problems.append(f"{rid}: response_hash does not match the response")
            if normalize_text(row["prompt_text"]) != payload.get("prompt", ""):
                report.problems.append(f"{rid}: prompt_text does not match the payload")
            if row["prev_chain_hash"] != expected_prev:
                report.problems.append(f"{rid}: chain broken (a row was removed or reordered)")

            if chain_digest(row) != row["chain_hash"]:
                report.problems.append(f"{rid}: chain_hash does not match its content")

            key = trusted_public_key_pem or row["public_key_pem"]
            if row["signature"] is None:
                report.problems.append(f"{rid}: not signed")
            elif not key or not verify_signature(key, row["chain_hash"], row["signature"]):
                report.problems.append(f"{rid}: invalid signature")

            expected_prev = row["chain_hash"]

        self._verify_annotations(report, rows, trusted_public_key_pem, anchors or [])

        heads = [row["chain_hash"] for row in rows]
        for anchor in anchors or []:
            n, when = int(anchor["records"]), anchor.get("created_at", "?")
            if n == 0:
                continue
            if n > len(heads):
                report.problems.append(
                    f"anchor of {when}: {n} records anchored, {len(heads)} left (records were removed)")
            elif heads[n - 1] != anchor["chain_hash"]:
                report.problems.append(
                    f"anchor of {when}: record {n} no longer has the anchored chain hash "
                    "(the history was rewritten)")
        return report

    def _verify_annotations(self, report: VerificationReport, interactions: list[sqlite3.Row],
                            trusted_public_key_pem: str | None, anchors: list[Mapping[str, Any]]) -> None:
        """Result hashes, chain, signatures and bindings of the annotations."""
        with self._connect() as conn:
            notes = conn.execute("SELECT * FROM annotations ORDER BY id").fetchall()
        targets = {row["id"]: row for row in interactions}
        expected_prev = GENESIS_HASH
        heads = []
        for note in notes:
            nid = f"annotation {note['id']}"
            report.annotations_checked += 1
            if sha256_hex(note["result_json"]) != note["result_hash"]:
                report.problems.append(f"{nid}: result_hash does not match the result")
            if "chain_hash" not in note.keys() or note["chain_hash"] is None:
                report.unchained_annotations += 1
                continue
            heads.append(note["chain_hash"])
            if note["prev_chain_hash"] != expected_prev:
                report.problems.append(f"{nid}: annotation chain broken (an annotation was removed or reordered)")
            if annotation_digest(note) != note["chain_hash"]:
                report.problems.append(f"{nid}: annotation chain_hash does not match its content")
            key = trusted_public_key_pem or note["public_key_pem"]
            if note["signature"] is None:
                report.problems.append(f"{nid}: not signed")
            elif not key or not verify_signature(key, note["chain_hash"], note["signature"]):
                report.problems.append(f"{nid}: invalid signature")
            target = targets.get(note["interaction_id"])
            if target is None:
                report.problems.append(f"{nid}: binding broken (the judged interaction is missing)")
            elif (target["chain_hash"] != note["target_chain_hash"]
                  or target["response_hash"] != note["target_response_hash"]):
                report.problems.append(f"{nid}: binding broken (the judged interaction was changed)")
            expected_prev = note["chain_hash"]
        if heads and report.unchained_annotations:
            report.problems.append(
                f"{report.unchained_annotations} annotation(s) without chain fields in a store whose "
                "annotations are chained")
        for anchor in anchors:
            if "annotations" not in anchor:
                continue
            n, when = int(anchor["annotations"]), anchor.get("created_at", "?")
            if n == 0:
                continue
            if n > len(heads):
                report.problems.append(
                    f"anchor of {when}: {n} annotations anchored, {len(heads)} left (annotations were removed)")
            elif heads[n - 1] != anchor["annotation_chain_hash"]:
                report.problems.append(
                    f"anchor of {when}: annotation {n} no longer has the anchored chain hash "
                    "(the annotation history was rewritten)")
