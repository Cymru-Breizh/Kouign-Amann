"""LLM provider abstraction using litellm for model-agnostic completions."""

import os

import litellm
import requests

HF_SERVER_URL = os.environ.get("HF_SERVER_URL", "http://hf-server:8000/v1")


def resolve_hf_model_id() -> str:
    """Query the HF inference server for the model it's actually serving."""
    try:
        resp = requests.get(f"{HF_SERVER_URL}/models", timeout=5)
        resp.raise_for_status()
        return resp.json()["data"][0]["id"]
    except Exception:
        return "unknown-hf-model"


def generate_response(
    model_id: str,
    system_message: str,
    user_message: str,
    temperature: float = 0,
    max_tokens: int = 500,
) -> str:
    """Call any LLM via litellm's unified interface.

    Model ID examples:
        OpenAI:    "gpt-4o", "gpt-4", "gpt-3.5-turbo"
        Anthropic: "anthropic/claude-sonnet-4-20250514"
        Ollama:    "ollama/llama3"
        Azure:     "azure/my-deployment"
        Fine-tuned: "ft:gpt-4o-2024-08-06:org:name:id"
        HF server: "hf/techiaith/llama-3.2-1b-welsh-sft"

    Models prefixed with "hf/" are routed to the local HF inference server
    (set HF_SERVER_URL env var, defaults to http://hf-server:8000/v1).

    See https://docs.litellm.ai/docs/providers for all supported providers.
    """
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    kwargs = dict(
        model=model_id,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if model_id.startswith("ollama/"):
        kwargs["think"] = False
        kwargs["num_ctx"] = 8192

    if model_id.startswith("hf/"):
        kwargs["model"] = model_id[3:]  # strip "hf/" prefix
        kwargs["api_base"] = HF_SERVER_URL
        kwargs["custom_llm_provider"] = "openai"
        kwargs["api_key"] = "none"

    response = litellm.completion(**kwargs)

    return response.choices[0].message.content
