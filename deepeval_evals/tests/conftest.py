import pytest


def pytest_addoption(parser):
    parser.addoption("--model", default="gpt-4o", help="LLM model ID (e.g. gpt-4o, anthropic/claude-sonnet-4-20250514, ollama/llama3)")
    parser.addoption("--max-samples", type=int, default=None, help="Limit number of eval samples")


@pytest.fixture
def model_id(request):
    return request.config.getoption("--model")


@pytest.fixture
def max_samples(request):
    return request.config.getoption("--max-samples")
