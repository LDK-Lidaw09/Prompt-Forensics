"""Thread-safe, crash-tolerant result writers shared by the batch runners.

Every row is flushed as soon as it is written, so an interrupted run loses at
most the rows in flight and can be resumed (``--resume``).
"""

from __future__ import annotations

import csv
import json
import os
import threading
from pathlib import Path
from typing import Any, Iterable


def _cell(value: Any) -> Any:
    """CSV cell: None becomes an empty cell, lists and dicts become JSON."""
    if value is None:
        return ""
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


class JsonlWriter:
    """Append JSON objects, one per line."""

    def __init__(self, path: str | Path, append: bool = True):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._file = open(path, "a" if append else "w", encoding="utf-8")

    def write(self, obj: dict[str, Any]) -> None:
        line = json.dumps(obj, ensure_ascii=False)
        with self._lock:
            self._file.write(line + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            self._file.close()

    def __enter__(self) -> "JsonlWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class CsvWriter:
    """Write dict rows with a fixed header. Appending keeps the existing header."""

    def __init__(self, path: str | Path, fieldnames: list[str], append: bool = False):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        has_content = os.path.exists(path) and os.path.getsize(path) > 0
        self._lock = threading.Lock()
        self._file = open(path, "a" if append else "w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=fieldnames)
        self.fieldnames = fieldnames
        if not (append and has_content):
            self._writer.writeheader()
            self._file.flush()

    def write(self, row: dict[str, Any]) -> None:
        clean = {k: _cell(row.get(k)) for k in self.fieldnames}
        with self._lock:
            self._writer.writerow(clean)
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            self._file.close()

    def __enter__(self) -> "CsvWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read_jsonl(paths: Iterable[str | Path]) -> Iterable[tuple[str, int, dict[str, Any]]]:
    """Yield ``(path, line_number, object)`` for every JSON line; bad lines raise."""
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        yield str(path), line_number, json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(f"{path}:{line_number}: invalid JSON ({exc})") from exc
