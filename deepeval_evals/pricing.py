"""Per-model token pricing for cost estimation.

Prices are in USD per 1,000,000 tokens (USD/MTok), matching how Anthropic
publishes them. Use ``lookup_pricing(model_id)`` to resolve a litellm-style
model ID (e.g. ``anthropic/claude-opus-4-7-20251101``) to its price entry.
"""

from __future__ import annotations

# Source of the Claude pricing table below.
CLAUDE_PRICING_SOURCE_URL = "https://platform.claude.com/docs/en/about-claude/pricing"
CLAUDE_PRICING_SNAPSHOT_DATE = "2026-05-01"

# Keyed by model "stem" — the part of the model ID before the date suffix and
# without the provider prefix. Match by longest-prefix to avoid the
# claude-opus-4 / claude-opus-4-1 collision.
#
# Fields are USD per 1M tokens:
#   input        — base input tokens (no caching)
#   cache_write_5m — writing to the 5-minute prompt cache
#   cache_write_1h — writing to the 1-hour prompt cache
#   cache_hit    — reading from cache (and refreshes)
#   output       — output tokens
CLAUDE_PRICING = {
    "claude-opus-4-7":   {"input":  5.00, "cache_write_5m":  6.25, "cache_write_1h": 10.0, "cache_hit": 0.50, "output": 25.0},
    "claude-opus-4-6":   {"input":  5.00, "cache_write_5m":  6.25, "cache_write_1h": 10.0, "cache_hit": 0.50, "output": 25.0},
    "claude-opus-4-5":   {"input":  5.00, "cache_write_5m":  6.25, "cache_write_1h": 10.0, "cache_hit": 0.50, "output": 25.0},
    "claude-opus-4-1":   {"input": 15.00, "cache_write_5m": 18.75, "cache_write_1h": 30.0, "cache_hit": 1.50, "output": 75.0},
    "claude-opus-4":     {"input": 15.00, "cache_write_5m": 18.75, "cache_write_1h": 30.0, "cache_hit": 1.50, "output": 75.0},
    "claude-sonnet-4-6": {"input":  3.00, "cache_write_5m":  3.75, "cache_write_1h":  6.0, "cache_hit": 0.30, "output": 15.0},
    "claude-sonnet-4-5": {"input":  3.00, "cache_write_5m":  3.75, "cache_write_1h":  6.0, "cache_hit": 0.30, "output": 15.0},
    "claude-sonnet-4":   {"input":  3.00, "cache_write_5m":  3.75, "cache_write_1h":  6.0, "cache_hit": 0.30, "output": 15.0},
    "claude-sonnet-3-7": {"input":  3.00, "cache_write_5m":  3.75, "cache_write_1h":  6.0, "cache_hit": 0.30, "output": 15.0, "deprecated": True},
    "claude-haiku-4-5":  {"input":  1.00, "cache_write_5m":  1.25, "cache_write_1h":  2.0, "cache_hit": 0.10, "output":  5.0},
    "claude-haiku-3-5":  {"input":  0.80, "cache_write_5m":  1.00, "cache_write_1h":  1.6, "cache_hit": 0.08, "output":  4.0},
    "claude-opus-3":     {"input": 15.00, "cache_write_5m": 18.75, "cache_write_1h": 30.0, "cache_hit": 1.50, "output": 75.0, "deprecated": True},
    "claude-haiku-3":    {"input":  0.25, "cache_write_5m":  0.30, "cache_write_1h":  0.5, "cache_hit": 0.03, "output":  1.25},
}


# Source of the OpenAI pricing table below.
OPENAI_PRICING_SOURCE_URL = "https://developers.openai.com/api/docs/pricing"
OPENAI_PRICING_SNAPSHOT_DATE = "2026-05-01"

# Keyed by model "stem" — matches via longest-prefix so dated/fine-tuned
# variants (e.g. ``gpt-5.4-mini-2025-04-16``, ``ft:gpt-5.4:org:...``) resolve
# to their base model. Only chat-completion models the eval runner can call
# are listed here; realtime, image, video, and audio-token pricing live in
# different shapes on the page and aren't relevant for this runner.
#
# Fields are USD per 1M tokens:
#   input         — base input tokens
#   cached_input  — input tokens served from prompt cache (None if not offered)
#   output        — output tokens
OPENAI_PRICING = {
    "gpt-5.5":             {"input":  5.00, "cached_input": 0.50,  "output":  30.00},
    "gpt-5.5-pro":         {"input": 30.00, "cached_input": None,  "output": 180.00},
    "gpt-5.4":             {"input":  2.50, "cached_input": 0.25,  "output":  15.00},
    "gpt-5.4-mini":        {"input":  0.75, "cached_input": 0.075, "output":   4.50},
    "gpt-5.4-nano":        {"input":  0.20, "cached_input": 0.02,  "output":   1.25},
    "gpt-5.4-pro":         {"input": 30.00, "cached_input": None,  "output": 180.00},
    "gpt-5.3-chat-latest": {"input":  1.75, "cached_input": 0.175, "output":  14.00},
    "gpt-5.3-codex":       {"input":  1.75, "cached_input": 0.175, "output":  14.00},
}


def _normalize_claude_stem(model_id: str) -> str:
    """Strip provider prefix, date suffix, and 1M-context marker."""
    name = model_id
    # litellm prefixes like "anthropic/..."
    if "/" in name:
        name = name.split("/", 1)[1]
    # 1M-context marker e.g. "claude-opus-4-7[1m]"
    if "[" in name:
        name = name.split("[", 1)[0]
    # date suffix e.g. "claude-sonnet-4-20250514" -> "claude-sonnet-4"
    parts = name.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) == 8:
        name = parts[0]
    return name


def _normalize_openai_stem(model_id: str) -> str:
    """Strip provider prefix and fine-tune wrapper from an OpenAI model ID."""
    name = model_id
    # Fine-tune wrapper: "ft:<base>:<org>:<name>:<id>" -> "<base>"
    if name.startswith("ft:"):
        parts = name.split(":")
        if len(parts) >= 2:
            name = parts[1]
    # litellm prefixes like "openai/..."
    if "/" in name:
        name = name.split("/", 1)[1]
    return name


def lookup_pricing(model_id: str) -> dict | None:
    """Return the pricing entry for ``model_id``, or None if unknown.

    Accepts litellm-style IDs: ``anthropic/claude-opus-4-7-20251101``,
    ``claude-sonnet-4-5``, ``claude-opus-4-7[1m]``, ``gpt-5.4-mini``,
    ``openai/gpt-5.4``, ``ft:gpt-5.4-mini-2025-04-16:org:name:id``, etc.
    Returns the underlying pricing dict (with extra ``model``, ``stem``,
    and ``provider`` keys filled in), or None for unknown models.
    """
    stem = _normalize_claude_stem(model_id)
    # Longest-prefix match so "claude-opus-4-1-..." beats "claude-opus-4".
    for key in sorted(CLAUDE_PRICING, key=len, reverse=True):
        if stem == key or stem.startswith(key + "-"):
            entry = dict(CLAUDE_PRICING[key])
            entry["model"] = model_id
            entry["stem"] = key
            entry["provider"] = "anthropic"
            return entry

    stem = _normalize_openai_stem(model_id)
    for key in sorted(OPENAI_PRICING, key=len, reverse=True):
        if stem == key or stem.startswith(key + "-"):
            entry = dict(OPENAI_PRICING[key])
            entry["model"] = model_id
            entry["stem"] = key
            entry["provider"] = "openai"
            return entry

    return None


def estimate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    pricing: dict,
    *,
    cached_input_tokens: int = 0,
) -> float:
    """Compute USD cost for a (input_tokens, output_tokens) usage given a pricing entry.

    ``cached_input_tokens`` is the subset of input tokens served from the prompt
    cache (priced at ``cache_hit``); the remainder is priced at ``input``. We
    don't model cache *writes* here since the eval runner doesn't write the
    cache today — when that changes, add a ``cache_write_tokens`` parameter.
    """
    fresh_input = max(input_tokens - cached_input_tokens, 0)
    return (
        fresh_input * pricing["input"]
        + cached_input_tokens * pricing["cache_hit"]
        + output_tokens * pricing["output"]
    ) / 1_000_000
