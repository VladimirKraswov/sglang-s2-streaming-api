from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from .runtime import StreamingRuntime
from .settings import load_settings

settings = load_settings()
logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))
runtime = StreamingRuntime(settings)


@asynccontextmanager
async def lifespan(_: FastAPI):
    await runtime.startup()
    yield
    await runtime.shutdown()


app = FastAPI(
    title="FishAudio S2 Streaming API",
    docs_url="/docs",
    redoc_url=None,
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


@app.exception_handler(ValueError)
async def value_error(_, exc: ValueError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(RuntimeError)
async def runtime_error(_, exc: RuntimeError):
    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.get("/healthz")
async def healthz():
    status = await runtime.status()
    if not status["ready"]:
        raise HTTPException(status_code=503, detail=status.get("detail") or "runtime is not ready")
    return {"status": "ok", **status}


@app.get("/status")
async def status():
    return await runtime.status()


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": settings.model_name,
                "object": "model",
                "created": 0,
                "owned_by": "sglang-omni",
                "root": settings.model_name,
            }
        ],
    }


@app.post("/v1/audio/stream")
async def wav_stream(payload: dict[str, Any]):
    await ensure_ready()
    planned = runtime.stream_payloads({**payload, "response_format": "wav"})
    headers = {
        "Cache-Control": "no-store",
        "X-Accel-Buffering": "no",
        "X-Target-First-Byte-Ms": str(settings.target_first_byte_ms),
        "X-Early-Wav-Header": "1" if settings.early_wav_header else "0",
        "X-TTFA-Text-Chunking": "1" if len(planned) > 1 else "0",
        "X-TTFA-Chunk-Count": str(len(planned)),
    }
    return StreamingResponse(
        runtime.wav_stream(payload),
        media_type="audio/wav",
        headers=headers,
    )


@app.post("/v1/audio/events")
async def sse_stream(payload: dict[str, Any]):
    await ensure_ready()
    return StreamingResponse(
        runtime.sse_stream(payload),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )


async def ensure_ready() -> None:
    if runtime.ready:
        return
    status_data = await runtime.status()
    if not status_data["ready"]:
        raise HTTPException(status_code=503, detail=status_data.get("detail") or "runtime is not ready")
