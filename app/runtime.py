from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, AsyncIterator

import httpx

from .audio import pcm_payload, silence_pcm, streaming_wav_header, wav_info
from .settings import Settings

logger = logging.getLogger(__name__)

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac"}


class StreamingRuntime:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._lock = asyncio.Lock()
        self._process: subprocess.Popen | None = None
        self._client: httpx.AsyncClient | None = None
        self._ready = False
        self._error = ""
        self._last_warmup_ms: float | None = None

    @property
    def ready(self) -> bool:
        return self._ready

    async def startup(self) -> None:
        async with self._lock:
            self._ready = False
            self._error = ""
            try:
                if self.settings.manage_backend:
                    await self._start_backend()
                else:
                    await self._wait_until_ready()
                self._client = httpx.AsyncClient(timeout=self._timeout())
                if self.settings.warmup_enabled:
                    await self._warmup()
                self._ready = True
            except Exception as exc:
                self._error = str(exc)
                logger.exception("Streaming runtime startup failed")
                await self._close_client()
                if self.settings.manage_backend:
                    await self._stop_backend()

    async def shutdown(self) -> None:
        async with self._lock:
            self._ready = False
            await self._close_client()
            await self._stop_backend()

    async def status(self) -> dict[str, Any]:
        ready, detail = await self._backend_ready()
        self._ready = ready
        if detail:
            self._error = detail
        return {
            "ready": ready,
            "engine": "sglang-omni-fishaudio-s2-streaming",
            "model_path": self.settings.model_path,
            "model_name": self.settings.model_name,
            "backend_url": self.settings.backend_url,
            "managed_backend": self.settings.manage_backend,
            "target_first_byte_ms": self.settings.target_first_byte_ms,
            "last_warmup_ms": round(self._last_warmup_ms, 1) if self._last_warmup_ms is not None else None,
            "streaming": {
                "early_wav_header": self.settings.early_wav_header,
                "text_chunking": self.settings.text_chunking_enabled,
                "first_chunk_words": self.settings.first_chunk_words,
                "chunk_words": self.settings.chunk_words,
                "join_silence_ms": self.settings.join_silence_ms,
                "sample_rate": self.settings.stream_sample_rate,
                "channels": self.settings.stream_channels,
                "bits_per_sample": self.settings.stream_bits_per_sample,
            },
            "defaults": self.default_request_params(),
            "detail": detail or self._error,
        }

    def default_request_params(self) -> dict[str, Any]:
        return {
            "voice": "default",
            "response_format": "wav",
            "speed": self.settings.speed,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "top_k": self.settings.top_k,
            "repetition_penalty": self.settings.repetition_penalty,
            "seed": self.settings.seed,
            "max_new_tokens": self.settings.max_new_tokens,
            "ttfa_text_chunking": self.settings.text_chunking_enabled,
            "first_chunk_words": self.settings.first_chunk_words,
            "chunk_words": self.settings.chunk_words,
        }

    async def wav_stream(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        started = time.perf_counter()
        first_audio_ms: float | None = None
        header_sent = False
        stream_payloads = self.stream_payloads({**payload, "response_format": "wav"})

        if self.settings.early_wav_header:
            yield streaming_wav_header(
                sample_rate=self.settings.stream_sample_rate,
                channels=self.settings.stream_channels,
                bits_per_sample=self.settings.stream_bits_per_sample,
            )
            header_sent = True

        for index, stream_payload in enumerate(stream_payloads):
            response = await self.open_sse_stream(stream_payload)
            async for chunk in self._sse_response_audio_chunks(response):
                if first_audio_ms is None:
                    first_audio_ms = (time.perf_counter() - started) * 1000
                    logger.info(
                        "First upstream audio chunk in %.1f ms | ttfa_chunk=%s/%s",
                        first_audio_ms,
                        index + 1,
                        len(stream_payloads),
                    )

                if not header_sent:
                    info = wav_info(chunk)
                    if info is None:
                        yield streaming_wav_header(
                            sample_rate=self.settings.stream_sample_rate,
                            channels=self.settings.stream_channels,
                            bits_per_sample=self.settings.stream_bits_per_sample,
                        )
                        yield chunk
                    else:
                        yield streaming_wav_header(
                            sample_rate=info.sample_rate,
                            channels=info.channels,
                            bits_per_sample=info.bits_per_sample,
                        )
                        yield info.payload
                    header_sent = True
                    continue

                audio = pcm_payload(chunk)
                if audio:
                    yield audio

            if index + 1 < len(stream_payloads) and self.settings.join_silence_ms:
                yield silence_pcm(
                    ms=self.settings.join_silence_ms,
                    sample_rate=self.settings.stream_sample_rate,
                    channels=self.settings.stream_channels,
                    bits_per_sample=self.settings.stream_bits_per_sample,
                )

    async def sse_stream(self, payload: dict[str, Any]) -> AsyncIterator[bytes]:
        response = await self.open_sse_stream(payload)
        try:
            async for chunk in response.aiter_bytes():
                if chunk:
                    yield chunk
        finally:
            await response.aclose()

    async def open_sse_stream(self, payload: dict[str, Any]) -> httpx.Response:
        request_payload = self.build_backend_request(payload, stream=True)
        request = self._http_client().build_request(
            "POST",
            f"{self.settings.backend_url}/v1/audio/speech",
            json=request_payload,
        )
        try:
            response = await self._http_client().send(request, stream=True)
        except httpx.HTTPError as exc:
            self._ready = False
            self._error = f"SGLang Omni backend is unreachable: {exc}"
            raise RuntimeError(self._error) from exc

        if response.status_code >= 400:
            detail = await self._async_error_detail(response)
            await response.aclose()
            raise RuntimeError(detail)
        return response

    def stream_payloads(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        text = str(payload.get("input") or payload.get("text") or "").strip()
        if not text:
            raise ValueError("Text must not be empty")

        enabled = bool(payload_value(payload, "ttfa_text_chunking", self.settings.text_chunking_enabled))
        first_words = int(payload_value(payload, "first_chunk_words", self.settings.first_chunk_words))
        chunk_words = int(payload_value(payload, "chunk_words", self.settings.chunk_words))
        if not enabled:
            return [payload]

        parts = split_for_ttfa(text, first_words=max(first_words, 1), chunk_words=max(chunk_words, 1))
        if len(parts) <= 1:
            return [payload]

        rows = []
        for index, part in enumerate(parts):
            row = dict(payload)
            row.pop("input", None)
            row["text"] = part
            row["ttfa_chunk_index"] = index
            row["ttfa_chunk_count"] = len(parts)
            rows.append(row)

        logger.info(
            "TTFA text chunking | chars=%s | chunks=%s | first_words=%s | chunk_words=%s",
            len(text),
            len(rows),
            first_words,
            chunk_words,
        )
        return rows

    def build_backend_request(self, payload: dict[str, Any], *, stream: bool) -> dict[str, Any]:
        text = str(payload.get("input") or payload.get("text") or "").strip()
        if not text:
            raise ValueError("Text must not be empty")

        response_format = str(payload_value(payload, "response_format", "wav")).lower().strip()
        if response_format != "wav":
            raise ValueError("Only response_format='wav' is supported")

        request: dict[str, Any] = {
            "model": payload.get("model") or self.settings.model_name,
            "input": text,
            "voice": str(payload_value(payload, "voice", "default")),
            "response_format": "wav",
            "speed": float(payload_value(payload, "speed", self.settings.speed)),
            "stream": stream,
            "temperature": float(payload_value(payload, "temperature", self.settings.temperature)),
            "top_p": float(payload_value(payload, "top_p", self.settings.top_p)),
            "top_k": int(payload_value(payload, "top_k", self.settings.top_k)),
            "repetition_penalty": float(
                payload_value(payload, "repetition_penalty", self.settings.repetition_penalty)
            ),
            "max_new_tokens": int(payload_value(payload, "max_new_tokens", self.settings.max_new_tokens)),
        }

        seed = payload.get("seed", self.settings.seed)
        if seed is not None:
            request["seed"] = int(seed)

        for key in ("language", "instructions", "task_type", "stage_params"):
            value = payload.get(key)
            if value is not None:
                request[key] = value

        references = self._build_references(payload)
        if references:
            request["references"] = references

        return request

    async def _start_backend(self) -> None:
        await self._stop_backend()
        if not self.settings.config_path.exists():
            raise FileNotFoundError(f"SGLang S2 config does not exist: {self.settings.config_path}")

        env = os.environ.copy()
        if self.settings.flashinfer_disable_version_check:
            env["FLASHINFER_DISABLE_VERSION_CHECK"] = "1"
        if self.settings.torchinductor_max_autotune:
            env["TORCHINDUCTOR_MAX_AUTOTUNE"] = "1"
        env["PYTORCH_CUDA_ALLOC_CONF"] = self.settings.pytorch_cuda_alloc_conf
        env["SGLANG_OMNI_STARTUP_TIMEOUT"] = str(self.settings.startup_timeout)

        command = self.settings.backend_command()
        logger.info("Starting SGLang Omni backend: %s", " ".join(command))
        self._process = subprocess.Popen(command, env=env)
        await self._wait_until_ready()

    async def _stop_backend(self) -> None:
        proc = self._process
        self._process = None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            await asyncio.to_thread(proc.wait, timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            await asyncio.to_thread(proc.wait, timeout=10)

    async def _close_client(self) -> None:
        client = self._client
        self._client = None
        if client is not None:
            await client.aclose()

    async def _wait_until_ready(self) -> None:
        deadline = time.monotonic() + max(self.settings.startup_timeout, 30)
        last_error = ""
        while time.monotonic() < deadline:
            proc = self._process
            if proc is not None and proc.poll() is not None:
                raise RuntimeError(f"SGLang Omni backend exited with code {proc.returncode}")
            ready, detail = await self._backend_ready()
            if ready:
                return
            last_error = detail
            await asyncio.sleep(2)
        raise RuntimeError(f"Timed out waiting for SGLang Omni backend: {last_error or 'no response'}")

    async def _backend_ready(self) -> tuple[bool, str]:
        proc = self._process
        if proc is not None and proc.poll() is not None:
            return False, f"SGLang Omni backend exited with code {proc.returncode}"
        try:
            client = self._client
            if client is None:
                async with httpx.AsyncClient(timeout=10) as probe_client:
                    response = await probe_client.get(f"{self.settings.backend_url}/v1/models")
            else:
                response = await client.get(f"{self.settings.backend_url}/v1/models", timeout=10)
            if response.status_code < 400:
                return True, ""
            return False, f"/v1/models returned {response.status_code}"
        except Exception as exc:
            return False, str(exc)

    async def _warmup(self) -> None:
        started = time.perf_counter()
        payload = {
            "input": self.settings.warmup_text,
            "response_format": "wav",
            "max_new_tokens": self.settings.warmup_max_new_tokens,
            "temperature": self.settings.temperature,
            "top_p": self.settings.top_p,
            "top_k": self.settings.top_k,
            "repetition_penalty": self.settings.repetition_penalty,
            "ttfa_text_chunking": False,
        }
        seen_audio = False
        for index in range(self.settings.warmup_requests):
            logger.info("Warmup streaming request %s/%s", index + 1, self.settings.warmup_requests)
            response = await self.open_sse_stream(payload)
            async for chunk in self._sse_response_audio_chunks(response):
                if chunk:
                    seen_audio = True
        self._last_warmup_ms = (time.perf_counter() - started) * 1000
        if not seen_audio:
            logger.warning("Warmup completed without audio chunks")
        logger.info("Warmup completed in %.1f ms", self._last_warmup_ms)

    async def _sse_response_audio_chunks(self, response: httpx.Response) -> AsyncIterator[bytes]:
        try:
            async for line in response.aiter_lines():
                data = sse_data(line)
                if data is None:
                    continue
                if data == "[DONE]":
                    break
                chunk = decode_sse_audio(data)
                if chunk:
                    yield chunk
        finally:
            await response.aclose()

    def _http_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout())
        return self._client

    def _timeout(self) -> httpx.Timeout:
        return httpx.Timeout(
            self.settings.request_timeout,
            connect=self.settings.connect_timeout,
            write=self.settings.connect_timeout,
            pool=self.settings.connect_timeout,
        )

    def _build_references(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        references: list[dict[str, Any]] = []

        reference_id = str(payload.get("reference_id") or "").strip()
        if reference_id:
            references.append(self._saved_reference(reference_id))

        ref_audio = payload.get("ref_audio")
        if ref_audio:
            ref_text = str(payload.get("ref_text") or "").strip()
            if not ref_text:
                raise ValueError("ref_text is required when ref_audio is provided")
            references.append({"audio_path": str(ref_audio), "text": ref_text})

        raw_refs = payload.get("references") or []
        if not isinstance(raw_refs, list):
            raise ValueError("references must be a list")

        for item in raw_refs:
            if not isinstance(item, dict):
                raise ValueError("Each reference must be an object")

            item_reference_id = str(item.get("reference_id") or "").strip()
            if item_reference_id:
                references.append(self._saved_reference(item_reference_id))
                continue

            vq_codes = item.get("vq_codes")
            text = str(item.get("text") or item.get("transcript") or item.get("ref_text") or "").strip()
            if vq_codes is not None:
                references.append({"vq_codes": vq_codes, "text": text})
                continue

            audio_path = (
                item.get("audio_path")
                or item.get("ref_audio")
                or item.get("audio_url")
                or item.get("url")
                or item.get("audio")
            )
            if audio_path is None:
                raise ValueError("Reference must include audio_path/ref_audio/audio_url/url/audio or vq_codes")
            if not text:
                raise ValueError("Reference must include text/transcript/ref_text")

            audio_value = str(audio_path).strip()
            if audio_value.startswith("data:"):
                raise ValueError("References must use local paths, HTTP URLs, or vq_codes; data URLs are not supported")
            references.append({"audio_path": audio_value, "text": text})

        return references

    def _saved_reference(self, reference_id: str) -> dict[str, str]:
        reference_dir = self.settings.reference_root / reference_id
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

    @staticmethod
    async def _async_error_detail(response: httpx.Response) -> str:
        content = await response.aread()
        try:
            data = json.loads(content.decode("utf-8", errors="replace"))
            return str(data.get("detail") or data)
        except Exception:
            return content.decode("utf-8", errors="replace") or f"Upstream returned {response.status_code}"


def payload_value(payload: dict[str, Any], key: str, default: Any) -> Any:
    value = payload.get(key)
    return default if value is None else value


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
    if not b64:
        return None
    return base64.b64decode(b64)
