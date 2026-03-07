"""
app/llm/client.py
─────────────────
LLM client abstraction.
Supports Ollama (local) and OpenAI-compatible endpoints.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.telemetry.logging_config import get_logger
from app.utils.config import get_config

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_llm() -> BaseChatModel:
    """Return a cached LLM instance based on configured provider."""
    cfg = get_config().llm

    if cfg.provider == "ollama":
        # Try new dedicated package first, fall back to community
        try:
            from langchain_ollama import ChatOllama  # type: ignore
        except ImportError:
            try:
                from langchain_community.chat_models import ChatOllama  # type: ignore
            except ImportError as e:
                raise RuntimeError(
                    "Ollama package missing. Run: pip install langchain-ollama"
                ) from e

        logger.info("llm_provider", provider="ollama", model=cfg.model)
        return ChatOllama(
            model=cfg.model,
            base_url=cfg.base_url,
            temperature=cfg.temperature,
        )

    elif cfg.provider == "openai":
        from langchain_openai import ChatOpenAI  # type: ignore
        logger.info("llm_provider", provider="openai", model=cfg.model)
        # `ChatOpenAI` constructor doesn't accept `max_tokens` directly, so map
        # the configured limit to `max_completion_tokens` instead.
        return ChatOpenAI(
            model=cfg.model,
            temperature=cfg.temperature,
            max_completion_tokens=cfg.max_tokens,
        )

    else:
        raise ValueError(
            f"Unsupported LLM provider: {cfg.provider!r}. Choose 'ollama' or 'openai'."
        )


def invoke_llm(system_prompt: str, user_prompt: str) -> str:
    """Simple wrapper: returns plain-text response string."""
    llm = get_llm()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    response = llm.invoke(messages)
    content = response.content
    return content.strip() if isinstance(content, str) else str(content).strip()
