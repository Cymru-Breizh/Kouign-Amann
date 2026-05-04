SHELL = /bin/sh
IMAGE = techiaith/welsh-evals
default: build

build:
	docker build --rm -t $(IMAGE) .

# Run all evals:        make eval MODEL=gpt-4o
# Run one eval:         make eval MODEL=gpt-4o EVAL=welsh-lexicon
# Limit samples:        make eval MODEL=gpt-4o MAX_SAMPLES=50
# Target a specific Ollama container (multi-GPU setup, see infra/ollama/):
#                       make eval MODEL=ollama/gemma4:26b OLLAMA=ollama-gpu3
# When OLLAMA is unset, OLLAMA_API_BASE from openai.env is used.
# Include extra evals from a sibling project (optional):
#                       make eval MODEL=gpt-4o EXTRA_EVALS_DIR=/abs/path/to/evals-cymraeg
# Run only the extras (skip the built-in evals):
#                       make eval MODEL=gpt-4o EVAL=extras EXTRA_EVALS_DIR=/abs/path/to/evals-cymraeg
# The container is left unnamed so multiple evals can run in parallel.
eval: build
	docker run --rm -it \
		--network llm-evals \
		--env-file=openai.env \
		$(if $(OLLAMA),-e OLLAMA_API_BASE=http://$(OLLAMA):11434) \
		$(if $(EXTRA_EVALS_DIR),-v $(EXTRA_EVALS_DIR):/app/extra-evals:ro -e EXTRA_EVALS_DIR=/app/extra-evals) \
		-v ${PWD}/results:/app/results \
		$(IMAGE) python -m deepeval_evals.run_all \
		--model $(MODEL) \
		$(if $(EVAL),--eval $(EVAL)) \
		$(if $(MAX_SAMPLES),--max-samples $(MAX_SAMPLES))

# Estimate token counts (and optional cost) without calling any LLM:
#   make estimate
#   make estimate EVAL=welsh-lexicon
#   make estimate MODEL=anthropic/claude-sonnet-4-5     # auto-fills Claude prices
#   make estimate PRICE_IN=3 PRICE_OUT=15               # manual USD per 1M tokens
#   make estimate EVAL=welsh-llm-texts MAX_SAMPLES=50
#   make estimate EXTRA_EVALS_DIR=/abs/path/to/evals-cymraeg   # include sibling-project evals
estimate: build
	docker run --rm -it \
		$(if $(EXTRA_EVALS_DIR),-v $(EXTRA_EVALS_DIR):/app/extra-evals:ro -e EXTRA_EVALS_DIR=/app/extra-evals) \
		$(IMAGE) python -m deepeval_evals.run_all --estimate \
		$(if $(MODEL),--model $(MODEL)) \
		$(if $(EVAL),--eval $(EVAL)) \
		$(if $(MAX_SAMPLES),--max-samples $(MAX_SAMPLES)) \
		$(if $(PRICE_IN),--price-in $(PRICE_IN)) \
		$(if $(PRICE_OUT),--price-out $(PRICE_OUT))

# Run pytest-style: make test MODEL=gpt-4o [OLLAMA=ollama-gpuN]
test: build
	docker run --rm -it \
		--network llm-evals \
		--env-file=openai.env \
		$(if $(OLLAMA),-e OLLAMA_API_BASE=http://$(OLLAMA):11434) \
		$(if $(EXTRA_EVALS_DIR),-v $(EXTRA_EVALS_DIR):/app/extra-evals:ro -e EXTRA_EVALS_DIR=/app/extra-evals) \
		-v ${PWD}/results:/app/results \
		$(IMAGE) pytest deepeval_evals/tests/ \
		--model $(MODEL) \
		$(if $(MAX_SAMPLES),--max-samples $(MAX_SAMPLES)) \
		-v

# Interactive shell for creating eval data
create: build
	docker run --rm -it --name techiaith-create-evals \
		--network llm-evals \
		--env-file=openai.env \
		-v ${PWD}/evals:/app/evals \
		-v ${PWD}/src:/app/src \
		-v ${PWD}/results:/app/results \
		$(IMAGE) bash

clean:
	-docker rmi $(IMAGE)
