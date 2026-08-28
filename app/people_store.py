from __future__ import annotations

import json
import re
import secrets
from pathlib import Path
from typing import Any

import numpy as np


DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "people"
INDEX = DATA_DIR / "people.json"
NAME_RE = re.compile(r"[^A-Za-zÀ-ÿ0-9 _-]+")


def _read() -> list[dict[str, Any]]:
    if not INDEX.exists():
        return []
    return json.loads(INDEX.read_text(encoding="utf-8"))


def _write(people: list[dict[str, Any]]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    temporary = INDEX.with_suffix(".tmp")
    temporary.write_text(json.dumps(people, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(INDEX)


def list_people() -> list[dict[str, Any]]:
    return [{key: value for key, value in person.items() if key != "embedding"} for person in _read()]


def get(person_id: str) -> dict[str, Any] | None:
    return next((person for person in _read() if person["id"] == person_id), None)


def enroll(jpeg: bytes, name: str, camera: int, embedding: list[float] | None = None) -> dict[str, Any]:
    people = _read()
    person_id = secrets.token_hex(8)
    safe_name = NAME_RE.sub("", name).strip() or f"Personne {len(people) + 1}"
    image_name = f"{person_id}.jpg"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / image_name).write_bytes(jpeg)
    person = {"id": person_id, "name": safe_name, "camera": camera, "image": image_name, "embedding": embedding}
    people.append(person)
    _write(people)
    return person


def match(embedding: list[float], threshold: float = 0.2) -> dict[str, Any] | None:
    vector = np.asarray(embedding, dtype=np.float32)
    for person in _read():
        stored = person.get("embedding")
        if not stored:
            continue
        score = float(np.dot(vector, np.asarray(stored, dtype=np.float32)))
        if score >= threshold:
            return person
    return None


def rename(person_id: str, name: str) -> dict[str, Any] | None:
    people = _read()
    for person in people:
        if person["id"] == person_id:
            person["name"] = NAME_RE.sub("", name).strip() or person["name"]
            _write(people)
            return person
    return None
