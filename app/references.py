from __future__ import annotations

from pathlib import Path
from typing import Any

from .payloads import compact_string, first_compact_string, first_truthy_value

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac"}
REFERENCE_AUDIO_FIELDS = ("audio_path", "ref_audio", "audio_url", "url", "audio")
REFERENCE_TEXT_FIELDS = ("text", "transcript", "ref_text")


def build_references(payload: dict[str, Any], reference_root: Path) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []

    reference_id = compact_string(payload.get("reference_id"))
    if reference_id:
        references.append(saved_reference(reference_root, reference_id))

    shortcut_reference = inline_reference(payload.get("ref_audio"), payload.get("ref_text"))
    if shortcut_reference is not None:
        references.append(shortcut_reference)

    raw_refs = payload.get("references") or []
    if not isinstance(raw_refs, list):
        raise ValueError("references must be a list")

    for item in raw_refs:
        if not isinstance(item, dict):
            raise ValueError("Each reference must be an object")
        references.append(reference_from_item(item, reference_root))

    return references


def reference_from_item(item: dict[str, Any], reference_root: Path) -> dict[str, Any]:
    reference_id = compact_string(item.get("reference_id"))
    if reference_id:
        return saved_reference(reference_root, reference_id)

    vq_codes = item.get("vq_codes")
    text = first_compact_string(item, REFERENCE_TEXT_FIELDS)
    if vq_codes is not None:
        return {"vq_codes": vq_codes, "text": text}

    audio_path = first_truthy_value(item, REFERENCE_AUDIO_FIELDS)
    if audio_path is None:
        raise ValueError("Reference must include audio_path/ref_audio/audio_url/url/audio or vq_codes")
    if not text:
        raise ValueError("Reference must include text/transcript/ref_text")
    return {"audio_path": reference_audio_path(audio_path), "text": text}


def saved_reference(reference_root: Path, reference_id: str) -> dict[str, str]:
    base_dir = reference_root.resolve()
    reference_dir = (base_dir / reference_id).resolve()
    try:
        reference_dir.relative_to(base_dir)
    except ValueError as exc:
        raise ValueError(f"Invalid reference_id: {reference_id}") from exc

    if not reference_dir.exists():
        raise ValueError(f"Reference does not exist: {reference_id}")

    transcript_path = reference_dir / "sample.lab"
    if not transcript_path.exists():
        raise ValueError(f"Reference transcript does not exist: {reference_id}")
    transcript = transcript_path.read_text(encoding="utf-8", errors="replace").strip()
    if not transcript:
        raise ValueError(f"Reference transcript is empty: {reference_id}")

    audio_path = reference_dir / "sample.wav"
    if not audio_path.exists():
        audio_path = next(
            (
                path
                for path in sorted(reference_dir.iterdir())
                if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
            ),
            None,
        )
    if audio_path is None:
        raise ValueError(f"Reference audio does not exist: {reference_id}")

    return {"audio_path": str(audio_path), "text": transcript}


def reference_audio_path(value: Any) -> str:
    audio_path = compact_string(value)
    if audio_path.startswith("data:"):
        raise ValueError("References must use local paths, HTTP URLs, or vq_codes; data URLs are not supported")
    return audio_path


def inline_reference(audio_path: Any, text: Any) -> dict[str, str] | None:
    if not audio_path:
        return None
    ref_text = compact_string(text)
    if not ref_text:
        raise ValueError("ref_text is required when ref_audio is provided")
    return {"audio_path": reference_audio_path(audio_path), "text": ref_text}
