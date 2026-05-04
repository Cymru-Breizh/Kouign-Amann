import os
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval_evals.loaders import load_jsonl_goldens
from deepeval_evals.metrics import WelshExactMatchMetric
from deepeval_evals.models import generate_response

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "evals",
    "welsh-mmlu-aligned", "data", "welsh-mmlu-aligned"
)
PRIVATE_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "evals-private",
    "welsh-mmlu-aligned", "data", "welsh-mmlu-aligned"
)


def _resolve(filename):
    """Check evals-private/ first for private files, fall back to evals/."""
    if "_private" in filename:
        private_path = os.path.join(PRIVATE_DATA_DIR, filename)
        if os.path.isfile(private_path):
            return private_path
    return os.path.join(DATA_DIR, filename)


def _run_eval(jsonl_path, model_id, max_samples):
    goldens = load_jsonl_goldens(jsonl_path, max_samples=max_samples)
    test_cases = []
    for g in goldens:
        actual = generate_response(model_id, g.system_message, g.user_message)
        test_cases.append(LLMTestCase(
            input=g.user_message,
            actual_output=actual,
            expected_output=g.expected_output,
        ))

    metric = WelshExactMatchMetric()
    evaluate(test_cases=test_cases, metrics=[metric])


def test_welsh_mmlu_aligned_public(model_id, max_samples):
    _run_eval(os.path.join(DATA_DIR, "samples_public.jsonl"), model_id, max_samples)


def test_welsh_mmlu_aligned_private(model_id, max_samples):
    _run_eval(_resolve("samples_private.jsonl"), model_id, max_samples)


# --- English parallel (baseline) ---

def test_welsh_mmlu_aligned_en_public(model_id, max_samples):
    _run_eval(os.path.join(DATA_DIR, "samples_en_public.jsonl"), model_id, max_samples)


def test_welsh_mmlu_aligned_en_private(model_id, max_samples):
    _run_eval(_resolve("samples_en_private.jsonl"), model_id, max_samples)
