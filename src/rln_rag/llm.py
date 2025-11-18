from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from .settings import Settings


def build_llm(settings: Settings) -> BaseChatModel:
    provider = settings.llm_provider.lower().strip()

    if provider == "langgraph":
        if not settings.langgraph_api_key:
            raise RuntimeError(
                "LANGGRAPH_API_KEY is required when LLM_PROVIDER=langgraph"
            )
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            api_key=settings.langgraph_api_key,
            base_url=settings.langgraph_base_url,
            model=settings.llm_model,
            temperature=settings.temperature,
        )

    if provider == "groq":
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        from langchain_groq import ChatGroq

        return ChatGroq(
            api_key=settings.groq_api_key,
            model_name=settings.llm_model,
            temperature=settings.temperature,
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.llm_model,
            temperature=settings.temperature,
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER '{settings.llm_provider}'. Use langgraph, groq, or ollama."
    )
