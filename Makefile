COMPOSE ?= docker compose
PYTHON ?= python3

-include .env
export

.PHONY: pull up down logs ps health profile check build

pull:
	$(COMPOSE) -f compose.yml pull

up:
	$(COMPOSE) -f compose.yml up -d

down:
	$(COMPOSE) -f compose.yml down

logs:
	$(COMPOSE) -f compose.yml logs -f

ps:
	$(COMPOSE) -f compose.yml ps

health:
	curl -fsS http://127.0.0.1:$${API_PORT:-7782}/healthz

profile:
	$(PYTHON) tools/measure_ttfa.py --url http://127.0.0.1:$${API_PORT:-7782}/v1/audio/stream --deadline-ms $${S2_TARGET_FIRST_BYTE_MS:-200}

check:
	PYTHONPYCACHEPREFIX=/tmp/sglang-s2-streaming-pycache $(PYTHON) -m py_compile $$(find app tools -name '*.py')
	$(PYTHON) -m unittest discover -s tests
	bash -n entrypoint.sh
	$(COMPOSE) -f compose.yml config >/dev/null

build:
	docker build --build-arg BASE_IMAGE=$${SGLANG_OMNI_IMAGE:-frankleeeee/sglang-omni:dev} -t $${LOCAL_IMAGE:-sglang-s2-streaming-api:local} .
