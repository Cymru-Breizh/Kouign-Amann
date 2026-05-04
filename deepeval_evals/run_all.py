#!/usr/bin/env python3
"""CLI runner for Welsh LLM evaluations.

Usage:
    python -m deepeval_evals.run_all --model gpt-4o
    python -m deepeval_evals.run_all --model anthropic/claude-sonnet-4-20250514 --eval welsh-lexicon
    python -m deepeval_evals.run_all --model ollama/llama3 --max-samples 50
    python -m deepeval_evals.run_all --model gpt-4o --eval welsh-legislation-translation
"""

import argparse
import csv
import os
import re
import sys
import time
from datetime import datetime

from deepeval.test_case import LLMTestCase
from tqdm import tqdm

from deepeval_evals.loaders import load_jsonl_goldens
from deepeval_evals.metrics.bleu_score import compute_corpus_bleu
from deepeval_evals.models import generate_response, resolve_hf_model_id
from deepeval_evals.pricing import (
    CLAUDE_PRICING,
    CLAUDE_PRICING_SNAPSHOT_DATE,
    CLAUDE_PRICING_SOURCE_URL,
    OPENAI_PRICING,
    OPENAI_PRICING_SNAPSHOT_DATE,
    OPENAI_PRICING_SOURCE_URL,
    lookup_pricing,
)

BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "evals")
PRIVATE_DIR = os.path.join(os.path.dirname(__file__), "..", "evals-private")

EVALS = {
    "welsh-llm-texts-public": {
        "jsonl": "welsh-llm-texts/data/welsh-llm-texts/samples_public.jsonl",
        "metric": "mcq",
        "max_tokens": 10,
    },
    "welsh-llm-texts-private": {
        "jsonl": "welsh-llm-texts/data/welsh-llm-texts/samples_private.jsonl",
        "metric": "mcq",
        "max_tokens": 10,
    },
    "welsh-mmlu-aligned-public": {
        "jsonl": "welsh-mmlu-aligned/data/welsh-mmlu-aligned/samples_public.jsonl",
        "metric": "mcq",
        "max_tokens": 10,
    },
    "welsh-mmlu-aligned-private": {
        "jsonl": "welsh-mmlu-aligned/data/welsh-mmlu-aligned/samples_private.jsonl",
        "metric": "mcq",
        "max_tokens": 10,
    },
    "welsh-mmlu-aligned-en-public": {
        "jsonl": "welsh-mmlu-aligned/data/welsh-mmlu-aligned/samples_en_public.jsonl",
        "metric": "mcq",
        "max_tokens": 10,
    },
    "welsh-mmlu-aligned-en-private": {
        "jsonl": "welsh-mmlu-aligned/data/welsh-mmlu-aligned/samples_en_private.jsonl",
        "metric": "mcq",
        "max_tokens": 10,
    },
}

# Generate brythonic-translation entries for all 9 directions x 2 splits (public/private only)
_BT_DIRECTIONS = ["br-cy", "fr-cy", "br-en", "br-fr", "cy-br", "cy-fr", "cy-en", "en-fr", "fr-en"]
_BT_SPLITS = [("-public", "_public"), ("-private", "_private")]
for _dir in _BT_DIRECTIONS:
    for _name_suffix, _file_suffix in _BT_SPLITS:
        EVALS[f"brythonic-translation-{_dir}{_name_suffix}"] = {
            "jsonl": f"brythonic-translation/data/brythonic-translation/samples_{_dir}{_file_suffix}.jsonl",
            "metric": "bleu",
            "max_tokens": 500,
        }


# Optional extra evals from a sibling project (e.g. llm-evals-cy-github).
# When EXTRA_EVALS_DIR points at a directory containing the eval folders below,
# they are registered under their original names. Missing dir = no-op.
_EXTRA_EVALS = {
    "welsh-lexicon": {
        "jsonl": "welsh-lexicon/data/welsh-lexicon/samples.jsonl",
        "metric": "exact_match",
    },
    "welsh-grammar": {
        "jsonl": "welsh-grammar/data/welsh-grammar/samples.jsonl",
        "metric": "exact_match",
    },
    "welsh-yes-no": {
        "jsonl": "welsh-yes-no/data/welsh-yes-no/samples.jsonl",
        "metric": "exact_match",
    },
    "welsh-obscenities": {
        "jsonl": "welsh-obscenities/data/welsh-obscenities/samples.jsonl",
        "metric": "exact_match",
    },
    "welsh-bilingual-placenames": {
        "jsonl": "welsh-bilingual-placenames/data/welsh-bilingual-placenames/samples.jsonl",
        "metric": "exact_match",
    },
    "welsh-legislation-translation": {
        "jsonl": "welsh-legislation-translation/data/welsh-legislation-translation/samples.jsonl",
        "metric": "bleu",
    },
    "welsh-registers": {
        "jsonl": "welsh-registers/data/welsh-registers/samples.jsonl",
        "metric": "exact_match",
    },
    "welsh-mmlu-lite": {
        "jsonl": "welsh-mmlu-lite/data/welsh-mmlu-lite/samples.jsonl",
        "metric": "mcq",
        "max_tokens": 10,
    },
    "welsh-toxigen": {
        "jsonl": "welsh-toxigen/data/welsh-toxigen/samples.jsonl",
        "metric": "exact_match",
    },
    "welsh-arc-easy-mini-cy": {
        "jsonl": "welsh-arc-easy-mini-cy/data/welsh-arc-easy-mini-cy/samples.jsonl",
        "metric": "mcq",
        "max_tokens": 10,
    },
}

def resolve_jsonl_path(config):
    """Resolve the JSONL path for an eval, checking evals-private/ first for private files."""
    jsonl = config["jsonl"]
    base = config.get("base_dir", BASE_DIR)
    if "_private" in jsonl:
        private_path = os.path.join(PRIVATE_DIR, jsonl)
        if os.path.isfile(private_path):
            return private_path
    return os.path.join(base, jsonl)


EXTRA_EVALS_DIR = os.environ.get("EXTRA_EVALS_DIR")
if EXTRA_EVALS_DIR and os.path.isdir(EXTRA_EVALS_DIR):
    for _name, _cfg in _EXTRA_EVALS.items():
        EVALS[_name] = {**_cfg, "base_dir": EXTRA_EVALS_DIR}


# Eval groups: shorthand names that expand to multiple evals
EVAL_GROUPS = {
    "brythonic-translation": [
        f"brythonic-translation-{d}{s}"
        for d in _BT_DIRECTIONS
        for s, _ in _BT_SPLITS
    ],
    "extras": list(_EXTRA_EVALS.keys()),
}

# Valid choices for --eval: individual evals + group names
EVAL_CHOICES = list(EVALS.keys()) + list(EVAL_GROUPS.keys())


def expand_eval_name(name):
    """Expand an eval name, resolving groups to their member evals."""
    if name in EVAL_GROUPS:
        members = EVAL_GROUPS[name]
        missing = [m for m in members if m not in EVALS]
        if missing:
            raise SystemExit(
                f"Eval group '{name}' includes {len(missing)} eval(s) that are not "
                f"registered: {', '.join(missing)}. Set EXTRA_EVALS_DIR to a directory "
                f"containing those eval folders and try again."
            )
        return members
    return [name]


def extract_mcq(text):
    """Extract first A/B/C/D letter from a response."""
    m = re.search(r'\b([A-D])\b', text)
    return m.group(1) if m else text.strip()


# Approx chat overhead per sample: role markers + assistant priming.
# Good enough for a pre-flight cost estimate; off by < ~10 tokens per sample.
CHAT_OVERHEAD_PER_SAMPLE = 7


# Realistic per-sample output token estimates by metric type.
# MCQ answers are a single letter (often with trailing whitespace/punctuation).
# exact_match answers are a short word or phrase.
# bleu (translation) outputs scale with input length, handled separately.
OUTPUT_TOKENS_PER_SAMPLE = {
    "mcq": 2,
    "exact_match": 8,
}


def estimate_eval(eval_name: str, encoding, max_samples: int = None):
    """Count input/output tokens for an eval without calling the LLM."""
    config = EVALS[eval_name]
    jsonl_path = resolve_jsonl_path(config)
    goldens = load_jsonl_goldens(jsonl_path, max_samples=max_samples)

    metric = config.get("metric")
    max_output_tokens = config.get("max_tokens", 500)
    input_tokens = 0
    output_tokens = 0
    for g in goldens:
        input_tokens += len(encoding.encode(g.system_message))
        input_tokens += len(encoding.encode(g.user_message))
        input_tokens += CHAT_OVERHEAD_PER_SAMPLE

        if metric == "bleu":
            # Translation output is approximately the same length as the user input.
            output_tokens += len(encoding.encode(g.user_message))
        elif metric in OUTPUT_TOKENS_PER_SAMPLE:
            output_tokens += OUTPUT_TOKENS_PER_SAMPLE[metric]
        else:
            output_tokens += max_output_tokens

    return {
        "eval": eval_name,
        "n": len(goldens),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    }


def run_estimate(eval_names, max_samples, price_in, price_out, model_id=None):
    import tiktoken
    encoding = tiktoken.get_encoding("o200k_base")

    print(f"Estimating tokens for {len(eval_names)} eval(s) using o200k_base tokenizer")
    print("(exact for gpt-4o family; ~10-15% approximation for Claude)\n")

    # If the user gave a model and didn't manually override prices, try to look
    # them up from the built-in pricing table.
    if model_id and (price_in is None or price_out is None):
        pricing = lookup_pricing(model_id)
        if pricing:
            if price_in is None:
                price_in = pricing["input"]
            if price_out is None:
                price_out = pricing["output"]
            print(
                f"Using pricing for {pricing['stem']}: "
                f"${pricing['input']}/MTok in, ${pricing['output']}/MTok out "
                f"(source: {CLAUDE_PRICING_SOURCE_URL}, snapshot {CLAUDE_PRICING_SNAPSHOT_DATE})\n"
            )
        else:
            print(f"No built-in pricing for '{model_id}'. Pass --price-in/--price-out to add cost.\n")

    rows = []
    for name in eval_names:
        try:
            rows.append(estimate_eval(name, encoding, max_samples=max_samples))
        except FileNotFoundError as e:
            print(f"Skipping {name}: file not found ({e.filename})")

    show_cost = price_in is not None and price_out is not None
    width = 82 if show_cost else 72

    if show_cost:
        header = f"{'Eval':<40} {'N':>6} {'Input tok':>12} {'Output tok':>12} {'Cost USD':>10}"
    else:
        header = f"{'Eval':<40} {'N':>6} {'Input tok':>12} {'Output tok':>12}"
    print(header)
    print("-" * width)

    total_in = total_out = total_n = 0
    for row in rows:
        total_in += row["input_tokens"]
        total_out += row["output_tokens"]
        total_n += row["n"]
        if show_cost:
            cost = (row["input_tokens"] * price_in + row["output_tokens"] * price_out) / 1_000_000
            print(f"{row['eval']:<40} {row['n']:>6} {row['input_tokens']:>12,} {row['output_tokens']:>12,} {f'${cost:.2f}':>10}")
        else:
            print(f"{row['eval']:<40} {row['n']:>6} {row['input_tokens']:>12,} {row['output_tokens']:>12,}")

    print("-" * width)
    if show_cost:
        total_cost = (total_in * price_in + total_out * price_out) / 1_000_000
        print(f"{'TOTAL':<40} {total_n:>6} {total_in:>12,} {total_out:>12,} {f'${total_cost:.2f}':>10}")
    else:
        print(f"{'TOTAL':<40} {total_n:>6} {total_in:>12,} {total_out:>12,}")

    print("\nOutput tokens are estimated per metric: ~2/sample for MCQ, ~8/sample for")
    print("exact-match, and ~input length for translation (BLEU). A verbose model may")
    print("exceed these; max_tokens (10 for MCQ, 500 for translation) is the hard cap.")
    if not show_cost:
        print("Pass --price-in and --price-out (USD per 1M tokens) to also estimate cost.")

    print_all_model_costs(total_in, total_out)


def print_all_model_costs(total_in: int, total_out: int):
    """Print estimated cost across every model in the built-in pricing tables."""
    _print_provider_costs("Claude", CLAUDE_PRICING,
                          CLAUDE_PRICING_SOURCE_URL, CLAUDE_PRICING_SNAPSHOT_DATE,
                          total_in, total_out)
    _print_provider_costs("OpenAI", OPENAI_PRICING,
                          OPENAI_PRICING_SOURCE_URL, OPENAI_PRICING_SNAPSHOT_DATE,
                          total_in, total_out)


def _print_provider_costs(provider: str, table: dict, source_url: str,
                          snapshot_date: str, total_in: int, total_out: int):
    print(f"\n{provider} pricing across all models (source: {source_url},")
    print(f"snapshot {snapshot_date}). Totals: "
          f"{total_in:,} input + {total_out:,} output tokens.\n")

    width = 86
    header = f"{'Model':<30} {'In $/MTok':>10} {'Out $/MTok':>11} {'Input $':>10} {'Output $':>10} {'Total $':>10}"
    print(header)
    print("-" * width)

    # Sort by total cost ascending so the cheapest is on top.
    rows = []
    for stem, p in table.items():
        cost_in = total_in * p["input"] / 1_000_000
        cost_out = total_out * p["output"] / 1_000_000
        rows.append((stem, p, cost_in, cost_out, cost_in + cost_out))
    rows.sort(key=lambda r: r[4])

    for stem, p, cost_in, cost_out, total_cost in rows:
        label = stem + (" (deprecated)" if p.get("deprecated") else "")
        in_rate = f"${p['input']:.2f}"
        out_rate = f"${p['output']:.2f}"
        c_in = f"${cost_in:.2f}"
        c_out = f"${cost_out:.2f}"
        c_total = f"${total_cost:.2f}"
        print(f"{label:<30} {in_rate:>10} {out_rate:>11} {c_in:>10} {c_out:>10} {c_total:>10}")
    print("-" * width)


def run_eval(eval_name: str, model_id: str, max_samples: int = None):
    config = EVALS[eval_name]
    jsonl_path = resolve_jsonl_path(config)

    print(f"\n{'='*60}")
    print(f"Eval: {eval_name}")
    print(f"Model: {model_id}")
    print(f"{'='*60}")

    goldens = load_jsonl_goldens(jsonl_path, max_samples=max_samples)
    print(f"Loaded {len(goldens)} samples")

    test_cases = []
    per_question = []
    predictions = []
    references = []

    max_tokens = config.get("max_tokens", 500)
    for i, g in enumerate(tqdm(goldens, desc="Generating responses")):
        actual = generate_response(model_id, g.system_message, g.user_message, max_tokens=max_tokens)
        predictions.append(actual)
        references.append(g.expected_output)
        test_cases.append(LLMTestCase(
            input=g.user_message,
            actual_output=actual,
            expected_output=g.expected_output,
        ))

        # Per-question result
        if config["metric"] == "mcq":
            is_correct = int(extract_mcq(actual) == g.expected_output.strip())
        elif config["metric"] == "exact_match":
            is_correct = int(actual.strip().strip(".,!?").lower() == g.expected_output.strip().strip(".,!?").lower())
        else:
            is_correct = ""

        per_question.append({
            "sample_id": g.sample_id,
            "correct": is_correct,
            "hypothesis": actual,
        })

    # Summary
    if config["metric"] in ("exact_match", "mcq"):
        correct = sum(r["correct"] for r in per_question)
        accuracy = correct / len(test_cases) * 100
        print(f"\nAccuracy: {accuracy:.2f}% ({correct}/{len(test_cases)})")
        return {
            "eval": eval_name, "metric": "accuracy",
            "score": f"{accuracy:.2f}", "n": len(test_cases),
            "per_question": per_question,
        }
    else:
        corpus_bleu = compute_corpus_bleu(predictions, references)
        print(f"\nCorpus BLEU: {corpus_bleu:.1f}")
        return {
            "eval": eval_name, "metric": "BLEU",
            "score": f"{corpus_bleu:.1f}", "n": len(test_cases),
            "per_question": per_question,
        }


def main():
    parser = argparse.ArgumentParser(description="Run Welsh LLM evaluations")
    parser.add_argument("--model", help="LLM model ID (e.g. gpt-4o, anthropic/claude-sonnet-4-20250514, ollama/llama3). Required unless --estimate is set.")
    parser.add_argument("--eval", choices=EVAL_CHOICES, help="Run a specific eval or group (default: all)")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of samples per eval")
    parser.add_argument("--estimate", action="store_true", help="Estimate input/output tokens per eval without calling the LLM")
    parser.add_argument("--price-in", type=float, help="USD per 1M input tokens (use with --estimate to also show cost)")
    parser.add_argument("--price-out", type=float, help="USD per 1M output tokens (use with --estimate to also show cost)")
    args = parser.parse_args()

    eval_names = expand_eval_name(args.eval) if args.eval else list(EVALS.keys())

    if args.estimate:
        run_estimate(eval_names, args.max_samples, args.price_in, args.price_out, model_id=args.model)
        return

    if not args.model:
        parser.error("--model is required unless --estimate is set")

    model_id = args.model
    if model_id.startswith("hf/"):
        actual_model = resolve_hf_model_id()
        print(f"HF server is serving: {actual_model}")
        model_id = f"hf/{actual_model}"

    print(f"Running {len(eval_names)} eval(s) with model: {model_id}")
    start = time.time()

    summaries = []
    for name in eval_names:
        summary = run_eval(name, model_id, args.max_samples)
        summaries.append(summary)

    elapsed = time.time() - start

    # Write results
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_slug = model_id.replace("/", "_")

    # Summary CSV
    csv_path = os.path.join(results_dir, f"{timestamp}_{model_slug}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["eval", "metric", "score", "n"])
        writer.writeheader()
        writer.writerows({k: v for k, v in s.items() if k != "per_question"} for s in summaries)

    # Per-question detail CSV
    detail_path = os.path.join(results_dir, f"{timestamp}_{model_slug}_detail.csv")
    with open(detail_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["eval", "sample_id", "correct", "hypothesis"])
        writer.writeheader()
        for s in summaries:
            for row in s["per_question"]:
                writer.writerow({"eval": s["eval"], **row})

    # Print summary table
    print(f"\n{'='*50}")
    print(f"{'Eval':<35} {'Metric':<10} {'Score':>8} {'N':>6}")
    print(f"{'-'*50}")
    for s in summaries:
        print(f"{s['eval']:<35} {s['metric']:<10} {s['score']:>8} {s['n']:>6}")
    print(f"{'='*50}")
    print(f"Summary saved to {csv_path}")
    print(f"Per-question detail saved to {detail_path}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
