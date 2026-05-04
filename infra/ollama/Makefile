SHELL = /bin/sh

# Which GPU(s) `make up` / `make down` act on. Override:
#   make up GPUS="0 3"
GPUS ?= 0

# Which single container the per-container targets act on (pull, list, gpu-check,
# test, logs). Defaults to the first GPU in $(GPUS). Override:
#   make pull GPU=3 MODEL=gemma4:26b
GPU ?= $(firstword $(GPUS))
CONTAINER = ollama-gpu$(GPU)

PROFILES = $(foreach g,$(GPUS),--profile gpu$(g))

default: up

# Bring up one Ollama container per GPU listed in $(GPUS).
up:
	docker compose $(PROFILES) up -d
	@echo
	@echo "Running (in-network DNS  ->  host port):"
	@for g in $(GPUS); do \
		printf "  ollama-gpu%s:11434  ->  localhost:%d\n" "$$g" "$$((11434+$$g))"; \
	done
	@echo
	@echo "Pull a model:    make pull GPU=$(GPU) MODEL=llama3"
	@echo "Run an eval:     (from repo root) make eval MODEL=ollama/llama3 OLLAMA=$(CONTAINER)"

# Stop only the containers in $(GPUS).
down:
	docker compose $(PROFILES) down

# Stop every ollama-gpu* container in this compose project, regardless of profile.
down-all:
	docker compose down --remove-orphans

logs:
	docker compose logs -f $(CONTAINER)

# Pull a model into one container: make pull GPU=3 MODEL=gemma4:26b
pull:
	docker exec $(CONTAINER) ollama pull $(MODEL)

# List models in one container (default $(CONTAINER)).
list:
	docker exec $(CONTAINER) ollama list

# List models in every running ollama-gpu* container.
list-all:
	@for c in $$(docker ps --filter "name=ollama-gpu" --format '{{.Names}}'); do \
		echo "=== $$c ==="; \
		docker exec $$c ollama list; \
	done

gpu-check:
	docker exec $(CONTAINER) nvidia-smi

# Quick liveness check against $(CONTAINER) on its host port (assumes llama3 is pulled there).
test:
	@port=$$((11434 + $(GPU))); \
	curl -s http://localhost:$$port/api/generate \
		-d '{"model":"llama3","prompt":"Say hello in Welsh","stream":false}' \
		| python3 -m json.tool

# Stop all ollama-gpu* containers and delete every model volume.
clean:
	docker compose down -v --remove-orphans
