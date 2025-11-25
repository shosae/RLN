"""LangGraph entrypoint using new app services."""

from __future__ import annotations

import os
from typing import TypedDict

from langgraph.runtime import Runtime

from app.config.settings import load_settings
from app.graph.graph_builder import build_orchestrator_graph
from app.services.llm_service import LLMConfig, build_llm
from app.services.executor_service import ExecutorService
from app.services.rag_service import RAGService, VectorStoreConfig
from app.services.robot_grpc_client import RobotGrpcClient

class Context(TypedDict, total=False):
    mode: str

_graph = None

def _build_graph():
    global _graph
    if _graph is not None:
        return _graph

    settings = load_settings()
    llm = build_llm(
        LLMConfig(
            provider=settings.llm_provider,
            model=settings.llm_model,
            temperature=settings.temperature,
            langgraph_api_key=os.getenv("LANGGRAPH_API_KEY"),
            langgraph_base_url=os.getenv("LANGGRAPH_BASE_URL"),
            groq_api_key=os.getenv("GROQ_API_KEY"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL"),
        )
    )

    rag_service = RAGService(
        VectorStoreConfig(
            docs_dir=settings.docs_dir,
            vectorstore_dir=settings.vectorstore_dir,
            embedding_model=settings.embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
    )
    retriever = rag_service.get_conversation_retriever(top_k=int(os.getenv("TOP_K", "4")))
    robot_client = RobotGrpcClient(settings.robot_grpc_target)
    executor = ExecutorService(robot_client=robot_client)

    _graph = build_orchestrator_graph(llm, retriever, executor)
    return _graph

def get_graph(runtime: Runtime[Context]):
    return _build_graph()

# 반드시 아래 라인을 추가해야 LangGraph 서버에서 인식합니다.
graph = _build_graph()
