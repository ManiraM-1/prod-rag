import re

import logfire
from portkey_ai import Portkey, createHeaders, PORTKEY_GATEWAY_URL
from langchain_openai import ChatOpenAI
from langchain_core.outputs import ChatResult

from app.config import settings


_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


def strip_reasoning(text: str) -> str:
    """
    Strip any leaked <think>...</think> block from a reasoning model's raw output.

    Groq's gpt-oss models are reasoning models: even with include_reasoning=False,
    the chain-of-thought can still leak into the response text instead of being
    suppressed or kept in a separate field. Any code that does exact/strict parsing
    of raw model output (e.g. checking `decision == "CONVERSATIONAL"`, or a
    yes/no verdict prefix) breaks silently if this leaks in unstripped — so this
    is applied centrally rather than left to each caller to remember.
    """
    return _THINK_BLOCK.sub("", text).strip()


class _NonReasoningChatOpenAI(ChatOpenAI):
    """ChatOpenAI subclass that strips leaked <think> blocks from every response."""

    def _generate(self, *args, **kwargs) -> ChatResult:
        result = super()._generate(*args, **kwargs)
        for generation in result.generations:
            generation.message.content = strip_reasoning(generation.message.content)
        return result


# Production gateway config:
#   - Fallback: primary @prod-rag/openai/gpt-oss-120b → @prod-rag1/openai/gpt-oss-20b on failure
#   - Cache: semantic mode (requires Portkey Enterprise — silently falls back to simple on free/starter)
#   - Retry: 2 attempts on rate limit / server error before triggering the fallback target
GATEWAY_CONFIG = {
    "strategy": {"mode": "fallback"},
    "cache": {"mode": "simple"},
    "retry": {
        "attempts": 2,
        "on_status_codes": [429, 503]
    },
    "targets": [
        {"override_params": {"model": f"@{settings.GROQ_SLUG}/openai/gpt-oss-120b"}},
        {"override_params": {"model": f"@{settings.GROQ_SLUG_2}/openai/gpt-oss-20b"}},
    ]
}

portkey_client = Portkey(
    api_key=settings.PORTKEY_API_KEY,
    # config=GATEWAY_CONFIG  # blocked: this org's keys require a saved dashboard Config, not inline JSON
).with_options(
    metadata={
        "feature": "prod-rag",
        "_user": "rag-system",
        "environment": "dev",
    }
)


def get_langchain_llm(
    feature: str = "prod-rag",
    model: str | None = None,
    temperature: float = 0,
    extra_body: dict | None = None,
) -> _NonReasoningChatOpenAI:
    """
    Returns a Portkey-backed ChatOpenAI, a drop-in for ChatGroq in LangChain nodes.

    Why ChatOpenAI and not ChatGroq:
      Portkey is a proxy. It exposes an OpenAI-compatible endpoint at PORTKEY_GATEWAY_URL.
      ChatGroq is hardwired to Groq's API and does not support routing through a proxy.
      ChatOpenAI supports base_url (points at Portkey) and default_headers (passes Portkey
      auth + config). The @prod-rag/model-name format is Portkey-specific, Groq's own client
      does not understand it. You are still using Groq models; Portkey is just in the middle.

    Every LLM call in this app — RAG pipeline or guardrails — goes through this
    one factory, so nothing bypasses Portkey's cost tracking, failover, and logs.
    `model`/`extra_body` let callers override the default model and pass extra
    provider-specific request fields (e.g. guardrails' reasoning-suppression
    params) without duplicating the client-construction logic elsewhere.
    """
    return _NonReasoningChatOpenAI(
        api_key=settings.PORTKEY_API_KEY,
        base_url=PORTKEY_GATEWAY_URL,
        model=model or f"@{settings.GROQ_SLUG}/openai/gpt-oss-120b",
        temperature=temperature,
        extra_body=extra_body or {},
        default_headers=createHeaders(
            api_key=settings.PORTKEY_API_KEY,
            # config=GATEWAY_CONFIG,  # blocked: this org's keys require a saved dashboard Config, not inline JSON
            metadata={
                "feature": feature,
                "_user": "rag-system",
                "environment": "dev"
            }
        )
    )

def extract_cache_status(response) -> str:
    """
    Pull x-portkey-cache-status from the Portkey native client response headers.
    Tries multiple attribute paths defensively — returns 'MISS' if not found.
    """
    for attr in ("_raw_response", "_response", "_http_response"):
        raw = getattr(response, attr, None)
        if raw is not None:
            status = getattr(raw, "headers", {}).get("x-portkey-cache-status", "")
            if status:
                return status.upper()
    return "MISS"
