from __future__ import annotations

import re
from typing import Any

FORWARDED_REQUEST_FIELDS = ("language", "instructions", "task_type", "stage_params")


def payload_value(payload: dict[str, Any], key: str, default: Any) -> Any:
    value = payload.get(key)
    return default if value is None else value


def compact_string(value: Any) -> str:
    return str(value or "").strip()


def request_text(payload: dict[str, Any]) -> str:
    text = compact_string(payload.get("input") or payload.get("text"))
    if not text:
        raise ValueError("Text must not be empty")
    return text


def ensure_wav_response_format(payload: dict[str, Any]) -> None:
    response_format = compact_string(payload_value(payload, "response_format", "wav")).lower()
    if response_format != "wav":
        raise ValueError("Only response_format='wav' is supported")


def first_truthy_value(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = payload.get(key)
        if value:
            return value
    return None


def first_compact_string(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    value = first_truthy_value(payload, keys)
    return compact_string(value)


def split_for_ttfa(text: str, *, first_words: int, chunk_words: int) -> list[str]:
    compact = re.sub(r"\s+", " ", text).strip()
    if not compact:
        return []

    words = compact.split(" ")
    if len(words) <= first_words + 2:
        return [compact]

    chunks = [" ".join(words[:first_words])]
    rest = words[first_words:]
    for pos in range(0, len(rest), chunk_words):
        chunks.append(" ".join(rest[pos : pos + chunk_words]))
    return [chunk for chunk in chunks if chunk]
