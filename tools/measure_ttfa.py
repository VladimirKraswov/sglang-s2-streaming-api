#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time

import httpx


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure first byte and first audio byte for /v1/audio/stream.")
    parser.add_argument("--url", default="http://127.0.0.1:7782/v1/audio/stream")
    parser.add_argument("--text", default="Привет, это проверка минимальной задержки первого аудио.")
    parser.add_argument("--deadline-ms", type=float, default=200.0)
    parser.add_argument("--audio-deadline-ms", type=float, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--no-chunking", action="store_true")
    args = parser.parse_args()

    payload = {
        "text": args.text,
        "ttfa_text_chunking": not args.no_chunking,
    }

    first_byte_ms = None
    first_audio_byte_ms = None
    received = 0
    started = time.perf_counter()

    with httpx.stream("POST", args.url, json=payload, timeout=args.timeout) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            if not chunk:
                continue
            now_ms = (time.perf_counter() - started) * 1000
            if first_byte_ms is None:
                first_byte_ms = now_ms
            received += len(chunk)
            if first_audio_byte_ms is None and received > 44:
                first_audio_byte_ms = now_ms
                break

    result = {
        "first_byte_ms": round(first_byte_ms or 0, 2),
        "first_audio_byte_ms": round(first_audio_byte_ms or 0, 2),
        "target_first_byte_ms": args.deadline_ms,
        "target_first_audio_byte_ms": args.audio_deadline_ms,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))

    failed = first_byte_ms is None or first_byte_ms > args.deadline_ms
    if args.audio_deadline_ms is not None:
        failed = failed or first_audio_byte_ms is None or first_audio_byte_ms > args.audio_deadline_ms
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
