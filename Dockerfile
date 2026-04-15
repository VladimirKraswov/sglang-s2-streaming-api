ARG BASE_IMAGE=frankleeeee/sglang-omni:dev
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    WORKSPACE=/workspace \
    HOST=0.0.0.0 \
    PORT=8888

WORKDIR /workspace

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install fastapi "uvicorn[standard]" httpx pydantic

COPY app /workspace/app
COPY config /workspace/config
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8888
ENTRYPOINT ["/entrypoint.sh"]
