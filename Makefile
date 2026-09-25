PY ?= .venv/Scripts/python
ifeq ($(OS),Windows_NT)
PY := .venv/Scripts/python
else
PY := .venv/bin/python
endif

.PHONY: venv install test lint chat ingest status docker-build docker-up docker-down docker-chat docker-ingest k8s-apply k8s-upload-docs k8s-ingest k8s-chat k8s-delete

venv:
	py -3.12 -m venv .venv || python3.12 -m venv .venv

install:
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check src tests

chat:
	$(PY) -m iu_agent chat

ingest:
	$(PY) -m iu_agent ingest

status:
	$(PY) -m iu_agent status

docker-build:
	docker compose build

docker-up:
	docker compose up -d qdrant

docker-down:
	docker compose down

docker-chat:
	docker compose run --rm agent

docker-ingest:
	docker compose run --rm agent ingest

k8s-apply:
	kubectl apply -k k8s/

k8s-upload-docs:
	bash scripts/k8s-upload-docs.sh

k8s-ingest:
	kubectl -n iu-agent delete job iu-agent-ingest --ignore-not-found
	kubectl apply -k k8s/
	kubectl -n iu-agent logs -f job/iu-agent-ingest

k8s-chat:
	kubectl -n iu-agent exec -it deploy/iu-agent -- iu-agent chat

k8s-delete:
	kubectl delete -k k8s/
