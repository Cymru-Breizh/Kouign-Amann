import os
from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval_evals.loaders import load_jsonl_goldens
from deepeval_evals.metrics.bleu_score import SacreBleuMetric
from deepeval_evals.models import generate_response

DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "evals",
    "brythonic-translation", "data", "brythonic-translation"
)
PRIVATE_DATA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "evals-private",
    "brythonic-translation", "data", "brythonic-translation"
)

DIRECTIONS = ["br-cy", "fr-cy", "br-en", "br-fr", "cy-br", "cy-fr", "cy-en", "en-fr", "fr-en"]


def _resolve(filename):
    """Check evals-private/ first for private files, fall back to evals/."""
    if "_private" in filename:
        private_path = os.path.join(PRIVATE_DATA_DIR, filename)
        if os.path.isfile(private_path):
            return private_path
    return os.path.join(DATA_DIR, filename)


def _run_eval(jsonl_filename, model_id, max_samples):
    jsonl_path = _resolve(jsonl_filename)
    goldens = load_jsonl_goldens(jsonl_path, max_samples=max_samples)
    test_cases = []
    for g in goldens:
        actual = generate_response(model_id, g.system_message, g.user_message, max_tokens=500)
        test_cases.append(LLMTestCase(
            input=g.user_message,
            actual_output=actual,
            expected_output=g.expected_output,
        ))

    metric = SacreBleuMetric()
    evaluate(test_cases=test_cases, metrics=[metric])


# --- Public splits ---

def test_brythonic_translation_br_cy_public(model_id, max_samples):
    _run_eval("samples_br-cy_public.jsonl", model_id, max_samples)

def test_brythonic_translation_fr_cy_public(model_id, max_samples):
    _run_eval("samples_fr-cy_public.jsonl", model_id, max_samples)

def test_brythonic_translation_br_en_public(model_id, max_samples):
    _run_eval("samples_br-en_public.jsonl", model_id, max_samples)

def test_brythonic_translation_br_fr_public(model_id, max_samples):
    _run_eval("samples_br-fr_public.jsonl", model_id, max_samples)

def test_brythonic_translation_cy_br_public(model_id, max_samples):
    _run_eval("samples_cy-br_public.jsonl", model_id, max_samples)

def test_brythonic_translation_cy_fr_public(model_id, max_samples):
    _run_eval("samples_cy-fr_public.jsonl", model_id, max_samples)

def test_brythonic_translation_cy_en_public(model_id, max_samples):
    _run_eval("samples_cy-en_public.jsonl", model_id, max_samples)

def test_brythonic_translation_en_fr_public(model_id, max_samples):
    _run_eval("samples_en-fr_public.jsonl", model_id, max_samples)

def test_brythonic_translation_fr_en_public(model_id, max_samples):
    _run_eval("samples_fr-en_public.jsonl", model_id, max_samples)


# --- Private splits ---

def test_brythonic_translation_br_cy_private(model_id, max_samples):
    _run_eval("samples_br-cy_private.jsonl", model_id, max_samples)

def test_brythonic_translation_fr_cy_private(model_id, max_samples):
    _run_eval("samples_fr-cy_private.jsonl", model_id, max_samples)

def test_brythonic_translation_br_en_private(model_id, max_samples):
    _run_eval("samples_br-en_private.jsonl", model_id, max_samples)

def test_brythonic_translation_br_fr_private(model_id, max_samples):
    _run_eval("samples_br-fr_private.jsonl", model_id, max_samples)

def test_brythonic_translation_cy_br_private(model_id, max_samples):
    _run_eval("samples_cy-br_private.jsonl", model_id, max_samples)

def test_brythonic_translation_cy_fr_private(model_id, max_samples):
    _run_eval("samples_cy-fr_private.jsonl", model_id, max_samples)

def test_brythonic_translation_cy_en_private(model_id, max_samples):
    _run_eval("samples_cy-en_private.jsonl", model_id, max_samples)

def test_brythonic_translation_en_fr_private(model_id, max_samples):
    _run_eval("samples_en-fr_private.jsonl", model_id, max_samples)

def test_brythonic_translation_fr_en_private(model_id, max_samples):
    _run_eval("samples_fr-en_private.jsonl", model_id, max_samples)
