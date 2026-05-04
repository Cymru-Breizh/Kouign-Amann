# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Welsh (Cymraeg) language evaluation suite for LLMs, built on DeepEval with LiteLLM for model-agnostic provider support. Tests models on Welsh lexicon recognition, legislation translation, yes/no questions, grammar, bilingual placenames, and obscenity detection.

The README and documentation are in English.

## Build & Run Commands

Everything runs in Docker:

```bash
make build                                          # Build Docker image
make eval MODEL=gpt-4o                              # Run all evals
make eval MODEL=gpt-4o EVAL=welsh-lexicon           # Run one eval
make eval MODEL=gpt-4o MAX_SAMPLES=50               # Limit samples
make eval MODEL=anthropic/claude-sonnet-4-20250514          # Use Anthropic
make eval MODEL=ollama/llama3                       # Use local Ollama
make test MODEL=gpt-4o                              # Run via pytest
make create                                         # Interactive shell for data generation
make clean                                          # Remove Docker image
```

Or without Docker:
```bash
pip install -r requirements.txt
python -m deepeval_evals.run_all --model gpt-4o --eval welsh-lexicon --max-samples 50
pytest deepeval_evals/tests/ --model gpt-4o --max-samples 50 -v
```

## Architecture

### Two-Phase Workflow

1. **Generate**: `src/{eval-name}/create_eval.py` scripts produce JSONL test data → public files to `evals/`, private files to `evals-private/`
2. **Evaluate**: `deepeval_evals/run_all.py` loads JSONL, calls LLMs via LiteLLM, scores with DeepEval metrics

### Key Directories

- `deepeval_evals/` — Evaluation runner, metrics, loaders, and pytest test files
  - `loaders/jsonl_loader.py` — Reads existing JSONL into Golden records
  - `metrics/exact_match.py` — Case-insensitive exact match (5 classification evals)
  - `metrics/bleu_score.py` — SacreBLEU scoring (translation eval)
  - `models.py` — LiteLLM wrapper for calling any LLM provider
  - `tests/` — One pytest file per eval, configured via `conftest.py`
  - `run_all.py` — CLI runner (alternative to pytest)
- `src/` — Private submodule (`Cymru-Breizh/Kouign-Amann-src`) with eval generation scripts and source data
- `evals/` — Generated JSONL test data (public splits)
- `evals-private/` — Private JSONL splits (git submodule → `Cymru-Breizh/Kouign-Amann-private`)
- `results/` — Evaluation output

### Eval Data Format

Each JSONL sample follows this structure:
```json
{"input": [{"role":"system","content":"..."}, {"role":"user","content":"..."}], "ideal": "expected_answer"}
```

Two metric types:
- **WelshExactMatchMetric**: Case-insensitive string match for Y/N classification and placename evals
- **SacreBleuMetric**: Per-sentence + corpus-level BLEU for the legislation translation eval

### Adding a New Eval

1. Create `src/{eval-name}/create_eval.py` in the private `Kouign-Amann-src` repo that outputs public JSONL to `evals/` and private JSONL to `evals-private/`
2. Run `make create` → `python3 create_eval.py` inside container
3. Add eval entry to `EVALS` dict in `deepeval_evals/run_all.py`
4. Create a test file in `deepeval_evals/tests/`
5. Run `make eval MODEL=gpt-4o EVAL={eval-name}` to test

### LLM Provider Support

Via LiteLLM, any provider is supported. Model ID format examples:
- OpenAI: `gpt-4o`, `gpt-4`, `ft:gpt-4o-2024-08-06:org:name:id`
- Anthropic: `anthropic/claude-sonnet-4-20250514`
- Ollama: `ollama/llama3`
- Azure: `azure/my-deployment`

## Environment

- Requires `openai.env` with API keys for the provider(s) you want to test
- Top-level `requirements.txt` has all dependencies (deepeval, litellm, sacrebleu, etc.)
- `src/requirements.txt` has additional dependencies for eval data generation (gensim, etc.)
