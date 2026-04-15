# FishAudio S2 Streaming API

Standalone streaming-only API for FishAudio S2-Pro on top of SGLang Omni.

This directory is intentionally self-contained and can be copied to a separate repository. It does not import or depend on the old gateway/render/live stack.

## Endpoints

- `GET /healthz` — readiness probe.
- `GET /status` — runtime settings and backend state.
- `GET /v1/models` — OpenAI-style model list.
- `POST /v1/audio/stream` — primary low-TTFA WAV stream.
- `POST /v1/audio/events` — raw upstream SSE stream with base64 WAV chunks.

There is no non-streaming synthesis endpoint by design.

## Quick Start

First pull the upstream image. It is very large, so run this inside `tmux` or with `nohup` on remote servers.

```bash
cp .env.example .env
make pull
```

Put the model under:

```text
./data/checkpoints/s2-pro
```

Then start:

```bash
make up
make logs
```

When ready:

```bash
make health
make profile
```

## Streaming WAV

```bash
curl -N -o /tmp/s2.wav \
  -H 'Content-Type: application/json' \
  -d '{"text":"Привет, это отдельный чистый API для потоковой озвучки FishAudio S2."}' \
  http://127.0.0.1:7782/v1/audio/stream
```

## Streaming SSE

```bash
curl -N \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello from standalone SGLang Omni S2.","response_format":"wav"}' \
  http://127.0.0.1:7782/v1/audio/events
```

## Request Fields

The API accepts a flexible JSON object:

- `text` or `input` — text to synthesize.
- `references` — list of `{ "audio_path": "...", "text": "..." }` or `{ "vq_codes": ..., "text": "..." }`.
- `ref_audio` + `ref_text` — single reference shortcut.
- `reference_id` — optional local reference from `S2_REFERENCE_ROOT/<id>/sample.wav` and `sample.lab`.
- `temperature`, `top_p`, `top_k`, `repetition_penalty`, `seed`, `max_new_tokens`, `speed`.
- `ttfa_text_chunking`, `first_chunk_words`, `chunk_words`.
- `language`, `instructions`, `task_type`, `stage_params`.

## TTFA Optimizations

Enabled by default:

- early WAV header before backend request;
- persistent local HTTP client to the SGLang backend;
- first text chunk of 5 words, next chunks of 7 words;
- full streaming warmup after startup;
- SGLang Omni CUDA graph decode remains enabled in the pipeline config;
- `stream_stride: 4`, `stream_followup_stride: 20`;
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

Useful settings in `.env`:

```bash
S2_EARLY_WAV_HEADER=true
S2_TEXT_CHUNKING=true
S2_FIRST_CHUNK_WORDS=5
S2_CHUNK_WORDS=7
S2_JOIN_SILENCE_MS=20
S2_WARMUP=true
S2_WARMUP_REQUESTS=1
S2_WARMUP_MAX_NEW_TOKENS=32
S2_TARGET_FIRST_BYTE_MS=200
```

For an external already-running `sgl-omni serve`, set:

```bash
S2_MANAGE_BACKEND=false
S2_BACKEND_URL=http://127.0.0.1:8092
```

## Notes

`/v1/audio/stream` returns a streaming WAV with a large placeholder RIFF data size. This is intentional for low first-byte latency. Most browsers, ffmpeg, and common clients can consume it as a live stream.

The text chunking strategy optimizes first audio arrival. It can slightly change prosody across chunk boundaries because the SGLang OpenAI-compatible TTS endpoint does not expose raw Fish Speech overlap/context controls.
