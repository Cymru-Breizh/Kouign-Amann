import sacrebleu
from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class SacreBleuMetric(BaseMetric):
    """Per-sentence BLEU score using sacrebleu.

    Note: BLEU is fundamentally a corpus-level metric. This computes
    sentence-level BLEU for DeepEval's per-test-case scoring. Use
    compute_corpus_bleu() to get the proper corpus-level score across
    all predictions.
    """

    def __init__(self, threshold: float = 0.0):
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.reason = ""

    @property
    def __name__(self):
        return "SacreBLEU"

    def measure(self, test_case: LLMTestCase) -> float:
        result = sacrebleu.sentence_bleu(
            test_case.actual_output,
            [test_case.expected_output],
        )
        self.score = result.score / 100.0  # normalize to 0-1
        self.success = self.score >= self.threshold
        self.reason = f"BLEU={result.score:.1f}"
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success


def compute_corpus_bleu(predictions: list[str], references: list[str]) -> float:
    """Compute corpus-level BLEU score across all predictions."""
    result = sacrebleu.corpus_bleu(predictions, [references])
    return result.score
