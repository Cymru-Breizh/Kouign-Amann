from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase


class WelshExactMatchMetric(BaseMetric):
    """Case-insensitive, whitespace-trimmed exact match metric."""

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.score = 0.0
        self.success = False
        self.reason = ""

    @property
    def __name__(self):
        return "Welsh Exact Match"

    def measure(self, test_case: LLMTestCase) -> float:
        actual = test_case.actual_output.strip().strip(".,!?").lower()
        expected = test_case.expected_output.strip().strip(".,!?").lower()
        self.score = 1.0 if actual == expected else 0.0
        self.success = self.score >= self.threshold
        self.reason = (
            f"Expected '{test_case.expected_output}', "
            f"got '{test_case.actual_output}'"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return self.success
