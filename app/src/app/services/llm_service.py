"""LLM 생성 및 래핑 로직."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from langchain_core.language_models.chat_models import BaseChatModel


@dataclass(slots=True)
class LLMConfig:
    provider: str
    model: str
    temperature: float = 0.2
    langgraph_api_key: Optional[str] = None
    langgraph_base_url: Optional[str] = None
    groq_api_key: Optional[str] = None
    ollama_base_url: Optional[str] = None


def build_llm(config: LLMConfig) -> BaseChatModel:
    """주어진 설정으로 Chat 모델을 생성한다."""

    provider = config.provider.lower().strip()

    if provider == "langgraph":
        if not config.langgraph_api_key:
            raise RuntimeError("LANGGRAPH_API_KEY가 필요합니다.")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=config.langgraph_api_key,
            base_url=config.langgraph_base_url,
            model=config.model,
            temperature=config.temperature,
        )

    if provider == "groq":
        if not config.groq_api_key:
            raise RuntimeError("GROQ_API_KEY가 필요합니다.")
        from langchain_groq import ChatGroq

        return ChatGroq(
            api_key=config.groq_api_key,
            model_name=config.model,
            temperature=config.temperature,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=config.ollama_base_url,
            model=config.model,
            temperature=config.temperature,
        )

    raise ValueError(
        f"지원하지 않는 LLM_PROVIDER '{config.provider}'."
    )
