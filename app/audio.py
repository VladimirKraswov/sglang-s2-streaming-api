from __future__ import annotations

from dataclasses import dataclass
import struct


@dataclass(frozen=True)
class WavInfo:
    channels: int
    sample_rate: int
    bits_per_sample: int
    payload: bytes


def streaming_wav_header(
    *,
    sample_rate: int,
    channels: int = 1,
    bits_per_sample: int = 16,
    data_size: int = 0x7FFFF000,
) -> bytes:
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    return b"".join(
        [
            b"RIFF",
            struct.pack("<I", min(36 + data_size, 0xFFFFFFFF)),
            b"WAVE",
            b"fmt ",
            struct.pack("<I", 16),
            struct.pack(
                "<HHIIHH",
                1,
                channels,
                sample_rate,
                byte_rate,
                block_align,
                bits_per_sample,
            ),
            b"data",
            struct.pack("<I", min(data_size, 0xFFFFFFFF)),
        ]
    )


def wav_info(data: bytes) -> WavInfo | None:
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        return None

    channels = 1
    sample_rate = 44100
    bits_per_sample = 16
    payload = b""
    pos = 12

    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        size = int.from_bytes(data[pos + 4 : pos + 8], "little")
        body_start = pos + 8
        body_end = body_start + size
        body = data[body_start:body_end]

        if chunk_id == b"fmt " and len(body) >= 16:
            channels = int.from_bytes(body[2:4], "little")
            sample_rate = int.from_bytes(body[4:8], "little")
            bits_per_sample = int.from_bytes(body[14:16], "little")
        elif chunk_id == b"data":
            payload = body
            break

        pos = body_end + (size % 2)

    return WavInfo(
        channels=channels,
        sample_rate=sample_rate,
        bits_per_sample=bits_per_sample,
        payload=payload,
    )


def pcm_payload(data: bytes) -> bytes:
    info = wav_info(data)
    return info.payload if info is not None else data


def silence_pcm(*, ms: int, sample_rate: int, channels: int, bits_per_sample: int) -> bytes:
    if ms <= 0:
        return b""
    bytes_per_sample = max(bits_per_sample // 8, 1)
    frames = int(sample_rate * ms / 1000)
    return b"\x00" * frames * channels * bytes_per_sample
