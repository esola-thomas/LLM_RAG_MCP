SHELL := /bin/bash
.DEFAULT_GOAL := help

ENV ?= .env
include $(ENV)
export

help:
	@echo "make up / down / ps / logs         - manage services"
	@echo "make pull-models                    - pull embedding model in ollama"
	@echo "make ingest DIR=./workspace         - ingest md/docx into Qdrant"
	@echo "make search Q='your query'          - test retrieval"
	@echo "make mcp                            - run MCP server locally"

up:
	docker compose -f infra/docker-compose.yml up -d

down:
	docker compose -f infra/docker-compose.yml down

ps:
	docker compose -f infra/docker-compose.yml ps

logs:
	docker compose -f infra/docker-compose.yml logs -f

pull-models:
	@echo "Pulling model: $(EMBED_MODEL)"
	curl -fsS -X POST "$(OLLAMA_URL)/api/pull" -H 'content-type: application/json' \
	  -d "{\"name\":\"$(EMBED_MODEL)\"}"

# --- Ingest/search via host ---
INGESTOR := apps/ingestor/.venv/bin/python apps/ingestor/src/ingest.py
SEARCHER := apps/ingestor/.venv/bin/python apps/ingestor/src/search.py

ingest:
	@if [[ -z "$(DIR)" ]]; then echo "Usage: make ingest DIR=./docs"; exit 1; fi
	$(INGESTOR) --path "$(DIR)" --corpus "$(DEFAULT_CORPUS)" \
	  --qdrant "$(QDRANT_URL)" --ollama "$(OLLAMA_URL)" --model "$(EMBED_MODEL)" \
	  --collection-prefix "$(COLLECTION_PREFIX)"

search:
	@if [[ -z "$(Q)" ]]; then echo "Usage: make search Q='question'"; exit 1; fi
	$(SEARCHER) --query "$(Q)" --corpus "$(DEFAULT_CORPUS)" \
	  --qdrant "$(QDRANT_URL)" --ollama "$(OLLAMA_URL)" --model "$(EMBED_MODEL)" \
	  --collection-prefix "$(COLLECTION_PREFIX)"

# --- MCP server (run on host) ---
mcp:
	cd apps/mcp-server && node dist/index.js
