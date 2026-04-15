from __future__ import annotations

import base64
import binascii
import json


def sse_data(line: str) -> str | None:
    if not line or not line.startswith("data:"):
        return None
    return line[len("data:") :].strip()


def decode_sse_audio(data: str) -> bytes | None:
    if data == "[DONE]":
        return None
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return None
    audio = payload.get("audio") or {}
    b64 = audio.get("data") if isinstance(audio, dict) else None
    if not isinstance(b64, str) or not b64:
        return None
    try:
        return base64.b64decode(b64)
    except (binascii.Error, ValueError):
        return None
