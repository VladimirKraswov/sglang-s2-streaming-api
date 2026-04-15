from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import shlex


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return float(value)


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    log_level: str

    model_path: str
    model_name: str
    config_path: Path
    reference_root: Path

    backend_url: str
    manage_backend: bool
    backend_host: str
    backend_port: int
    backend_log_level: str
    sgl_omni_bin: str
    sgl_omni_extra_args: str
    startup_timeout: int
    request_timeout: int
    connect_timeout: int

    warmup_enabled: bool
    warmup_requests: int
    warmup_text: str
    warmup_max_new_tokens: int

    target_first_byte_ms: int
    early_wav_header: bool
    text_chunking_enabled: bool
    first_chunk_words: int
    chunk_words: int
    join_silence_ms: int

    stream_sample_rate: int
    stream_channels: int
    stream_bits_per_sample: int

    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    seed: int | None
    max_new_tokens: int
    speed: float

    flashinfer_disable_version_check: bool
    torchinductor_max_autotune: bool
    pytorch_cuda_alloc_conf: str

    def backend_command(self) -> list[str]:
        command = [
            self.sgl_omni_bin,
            "serve",
            "--model-path",
            self.model_path,
            "--config",
            str(self.config_path),
            "--host",
            self.backend_host,
            "--port",
            str(self.backend_port),
            "--model-name",
            self.model_name,
            "--log-level",
            self.backend_log_level,
        ]
        if self.sgl_omni_extra_args.strip():
            command.extend(shlex.split(self.sgl_omni_extra_args))
        return command


def load_settings() -> Settings:
    backend_host = os.getenv("S2_BACKEND_HOST", "127.0.0.1")
    backend_port = _int_env("S2_BACKEND_PORT", 8092)
    backend_url = os.getenv("S2_BACKEND_URL", f"http://{backend_host}:{backend_port}").rstrip("/")
    seed = os.getenv("S2_SEED") or os.getenv("SEED")

    return Settings(
        host=os.getenv("HOST", "0.0.0.0"),
        port=_int_env("PORT", 8888),
        log_level=os.getenv("LOG_LEVEL", "info"),
        model_path=os.getenv("S2_MODEL_PATH", "/data/checkpoints/s2-pro"),
        model_name=os.getenv("S2_MODEL_NAME", "fishaudio-s2-streaming"),
        config_path=Path(os.getenv("S2_CONFIG_PATH", "/workspace/config/s2pro_streaming.yaml")),
        reference_root=Path(os.getenv("S2_REFERENCE_ROOT", "/data/references")),
        backend_url=backend_url,
        manage_backend=_bool_env("S2_MANAGE_BACKEND", os.getenv("S2_BACKEND_URL") is None),
        backend_host=backend_host,
        backend_port=backend_port,
        backend_log_level=os.getenv("S2_BACKEND_LOG_LEVEL", "info"),
        sgl_omni_bin=os.getenv("SGL_OMNI_BIN", "sgl-omni"),
        sgl_omni_extra_args=os.getenv("S2_EXTRA_ARGS", ""),
        startup_timeout=_int_env("S2_STARTUP_TIMEOUT", 2400),
        request_timeout=_int_env("S2_REQUEST_TIMEOUT", 3600),
        connect_timeout=_int_env("S2_CONNECT_TIMEOUT", 30),
        warmup_enabled=_bool_env("S2_WARMUP", True),
        warmup_requests=max(_int_env("S2_WARMUP_REQUESTS", 1), 0),
        warmup_text=os.getenv("S2_WARMUP_TEXT", "Warm up the low latency streaming path."),
        warmup_max_new_tokens=_int_env("S2_WARMUP_MAX_NEW_TOKENS", 32),
        target_first_byte_ms=_int_env("S2_TARGET_FIRST_BYTE_MS", 200),
        early_wav_header=_bool_env("S2_EARLY_WAV_HEADER", True),
        text_chunking_enabled=_bool_env("S2_TEXT_CHUNKING", True),
        first_chunk_words=max(_int_env("S2_FIRST_CHUNK_WORDS", 5), 1),
        chunk_words=max(_int_env("S2_CHUNK_WORDS", 7), 1),
        join_silence_ms=max(_int_env("S2_JOIN_SILENCE_MS", 20), 0),
        stream_sample_rate=_int_env("S2_STREAM_SAMPLE_RATE", 44100),
        stream_channels=_int_env("S2_STREAM_CHANNELS", 1),
        stream_bits_per_sample=_int_env("S2_STREAM_BITS_PER_SAMPLE", 16),
        temperature=_float_env("S2_TEMPERATURE", 0.8),
        top_p=_float_env("S2_TOP_P", 0.8),
        top_k=_int_env("S2_TOP_K", 30),
        repetition_penalty=_float_env("S2_REPETITION_PENALTY", 1.1),
        seed=int(seed) if seed else None,
        max_new_tokens=_int_env("S2_MAX_NEW_TOKENS", 1024),
        speed=_float_env("S2_SPEED", 1.0),
        flashinfer_disable_version_check=_bool_env("FLASHINFER_DISABLE_VERSION_CHECK", True),
        torchinductor_max_autotune=_bool_env("TORCHINDUCTOR_MAX_AUTOTUNE", True),
        pytorch_cuda_alloc_conf=os.getenv("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"),
    )
