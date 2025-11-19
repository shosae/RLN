"""RAG/vectorstore 로딩과 검색 서비스."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rln_rag.vectorstore import (
    load_documents,
    split_documents,
    build_embeddings,
)


@dataclass(slots=True)
class VectorStoreConfig:
    docs_dir: Path
    vectorstore_dir: Path
    embedding_model: str


def _build_vectorstore(config: VectorStoreConfig) -> FAISS:
    docs = load_documents(config)
    split_docs = split_documents(docs, config)
    embeddings = build_embeddings(config)
    store = FAISS.from_documents(split_docs, embeddings)
    config.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(config.vectorstore_dir))
    return store


def load_vectorstore(config: VectorStoreConfig) -> FAISS:
    embeddings = build_embeddings(config)
    index_file = config.vectorstore_dir / "index.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            f"Vector store not found at {config.vectorstore_dir}."
        )
    return FAISS.load_local(
        str(config.vectorstore_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )


class RAGService:
    """Conversation/waypoint retriever 묶음."""

    def __init__(
        self,
        conversation_config: VectorStoreConfig,
        waypoint_config: Optional[VectorStoreConfig] = None,
    ) -> None:
        self.conversation_config = conversation_config
        self.waypoint_config = waypoint_config
        self._conversation_retriever: Optional[BaseRetriever] = None
        self._waypoint_retriever: Optional[BaseRetriever] = None

    def get_conversation_retriever(self, top_k: int) -> BaseRetriever:
        if self._conversation_retriever is None:
            store = load_vectorstore(self.conversation_config)
            self._conversation_retriever = store.as_retriever(
                search_kwargs={"k": top_k}
            )
        return self._conversation_retriever

    def get_waypoint_retriever(self, top_k: int) -> Optional[BaseRetriever]:
        if not self.waypoint_config:
            return None
        if self._waypoint_retriever is None:
            store = load_vectorstore(self.waypoint_config)
            self._waypoint_retriever = store.as_retriever(
                search_kwargs={"k": top_k}
            )
        return self._waypoint_retriever
