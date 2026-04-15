ARG BASE_IMAGE=frankleeeee/sglang-omni:dev
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WORKSPACE=/workspace \
    HOST=0.0.0.0 \
    PORT=8888 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends git python3-venv \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv --system-site-packages "$VIRTUAL_ENV" \
    && "$VIRTUAL_ENV/bin/pip" install --upgrade pip setuptools wheel

RUN "$VIRTUAL_ENV/bin/pip" install fastapi "uvicorn[standard]" httpx pydantic

RUN git clone --depth 1 https://github.com/sgl-project/sglang-omni.git /opt/sglang-omni \
    && "$VIRTUAL_ENV/bin/pip" install -v /opt/sglang-omni

COPY app /workspace/app
COPY config /workspace/config
COPY entrypoint.sh /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh

EXPOSE 8888
ENTRYPOINT ["/workspace/entrypoint.sh"]