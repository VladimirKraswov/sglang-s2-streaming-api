from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
import tempfile
import unittest

import httpx

from app.payloads import request_text, split_for_ttfa
from app.references import inline_reference
from app.runtime import StreamingRuntime
from app.settings import Settings
from app.sse import decode_sse_audio, sse_data


def runtime_with(**overrides) -> StreamingRuntime:
    values = {
        "host": "0.0.0.0",
        "port": 8888,
        "log_level": "info",
        "model_path": "/data/checkpoints/s2-pro",
        "model_name": "fishaudio-s2-streaming",
        "config_path": Path("/workspace/config/s2pro_streaming.yaml"),
        "reference_root": Path("/data/references"),
        "backend_url": "http://127.0.0.1:8092",
        "manage_backend": False,
        "backend_host": "127.0.0.1",
        "backend_port": 8092,
        "backend_log_level": "info",
        "sgl_omni_bin": "sgl-omni",
        "sgl_omni_extra_args": "",
        "startup_timeout": 30,
        "request_timeout": 60,
        "connect_timeout": 5,
        "warmup_enabled": False,
        "warmup_requests": 0,
        "warmup_text": "warm up",
        "warmup_max_new_tokens": 32,
        "target_first_byte_ms": 200,
        "early_wav_header": True,
        "text_chunking_enabled": True,
        "first_chunk_words": 5,
        "chunk_words": 7,
        "join_silence_ms": 20,
        "stream_sample_rate": 44100,
        "stream_channels": 1,
        "stream_bits_per_sample": 16,
        "temperature": 0.8,
        "top_p": 0.8,
        "top_k": 30,
        "repetition_penalty": 1.1,
        "seed": None,
        "max_new_tokens": 1024,
        "speed": 1.0,
        "flashinfer_disable_version_check": True,
        "torchinductor_max_autotune": True,
        "pytorch_cuda_alloc_conf": "expandable_segments:True",
    }
    values.update(overrides)
    return StreamingRuntime(Settings(**values))


class RuntimeHelperTests(unittest.TestCase):
    def test_request_text_prefers_input_and_strips_whitespace(self) -> None:
        self.assertEqual(request_text({"input": "  hello  ", "text": "fallback"}), "hello")

        with self.assertRaisesRegex(ValueError, "Text must not be empty"):
            request_text({"text": "   "})

    def test_split_for_ttfa_compacts_whitespace_and_chunks_words(self) -> None:
        self.assertEqual(
            split_for_ttfa(" one   two\nthree four five six seven ", first_words=2, chunk_words=2),
            ["one two", "three four", "five six", "seven"],
        )

    def test_stream_payloads_rewrites_input_chunks_to_text(self) -> None:
        runtime = runtime_with(text_chunking_enabled=True, first_chunk_words=2, chunk_words=2)

        payloads = runtime.stream_payloads({"input": "one two three four five six", "voice": "demo"})

        self.assertEqual([payload["text"] for payload in payloads], ["one two", "three four", "five six"])
        self.assertNotIn("input", payloads[0])
        self.assertEqual(payloads[0]["ttfa_chunk_index"], 0)
        self.assertEqual(payloads[-1]["ttfa_chunk_count"], 3)
        self.assertEqual(payloads[-1]["voice"], "demo")

    def test_build_backend_request_normalizes_options_and_references(self) -> None:
        runtime = runtime_with(model_name="test-model", seed=None)

        request = runtime.build_backend_request(
            {
                "text": "  hello  ",
                "seed": "42",
                "language": "en",
                "references": [
                    {
                        "audio_path": "",
                        "audio_url": "http://example.test/ref.wav",
                        "text": "",
                        "transcript": "voice text",
                    }
                ],
            },
            stream=True,
        )

        self.assertEqual(request["model"], "test-model")
        self.assertEqual(request["input"], "hello")
        self.assertEqual(request["response_format"], "wav")
        self.assertEqual(request["seed"], 42)
        self.assertEqual(request["language"], "en")
        self.assertEqual(
            request["references"],
            [{"audio_path": "http://example.test/ref.wav", "text": "voice text"}],
        )

    def test_saved_reference_falls_back_to_supported_audio_file(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            reference_dir = Path(root) / "voice-a"
            reference_dir.mkdir()
            (reference_dir / "sample.lab").write_text("reference transcript\n", encoding="utf-8")
            (reference_dir / "sample.flac").write_bytes(b"fake-audio")

            runtime = runtime_with(reference_root=Path(root))
            request = runtime.build_backend_request({"text": "hello", "reference_id": "voice-a"}, stream=True)

        self.assertEqual(request["references"][0]["text"], "reference transcript")
        self.assertTrue(request["references"][0]["audio_path"].endswith("sample.flac"))

    def test_inline_reference_rejects_data_urls(self) -> None:
        with self.assertRaisesRegex(ValueError, "data URLs"):
            inline_reference("data:audio/wav;base64,AAAA", "voice text")

    def test_sse_helpers_extract_audio_bytes(self) -> None:
        audio = b"abc123"
        payload = '{"audio":{"data":"%s"}}' % base64.b64encode(audio).decode("ascii")

        self.assertEqual(sse_data("data: [DONE]"), "[DONE]")
        self.assertEqual(decode_sse_audio(payload), audio)
        self.assertIsNone(decode_sse_audio("not json"))
        self.assertIsNone(decode_sse_audio('{"audio":{"data":"%%%invalid%%%"}}'))

    def test_wav_stream_decodes_audio_from_mocked_sse_backend(self) -> None:
        async def run() -> tuple[list[bytes], dict[str, object]]:
            audio = b"\x01\x02\x03\x04"
            seen_request: dict[str, object] = {}

            async def handler(request: httpx.Request) -> httpx.Response:
                seen_request.update(json.loads(request.content.decode("utf-8")))
                payload = {
                    "audio": {
                        "data": base64.b64encode(audio).decode("ascii"),
                    }
                }
                content = f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n".encode("utf-8")
                return httpx.Response(200, content=content, headers={"content-type": "text/event-stream"})

            runtime = runtime_with(early_wav_header=True, text_chunking_enabled=False)
            runtime._client = httpx.AsyncClient(transport=httpx.MockTransport(handler), timeout=runtime._timeout())
            try:
                chunks = [chunk async for chunk in runtime.wav_stream({"text": "hello"})]
            finally:
                await runtime.shutdown()
            return chunks, seen_request

        chunks, seen_request = asyncio.run(run())

        self.assertEqual(len(chunks), 2)
        self.assertTrue(chunks[0].startswith(b"RIFF"))
        self.assertEqual(chunks[1], b"\x01\x02\x03\x04")
        self.assertEqual(seen_request["input"], "hello")
        self.assertIs(seen_request["stream"], True)


if __name__ == "__main__":
    unittest.main()
