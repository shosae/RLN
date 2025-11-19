"""Vectorstore helper functions."""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS

from rln_rag.vectorstore import (
    load_documents,
    split_documents,
    build_embeddings,
)


def build_vectorstore(docs_dir: Path, vectorstore_dir: Path, embedding_model: str) -> FAISS:
    docs = load_documents(docs_dir)
    chunks = split_documents(docs, docs_dir)
    embeddings = build_embeddings(embedding_model)
    store = FAISS.from_documents(chunks, embeddings)
    vectorstore_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(vectorstore_dir))
    return store


def load_vectorstore(docs_dir: Path, vectorstore_dir: Path, embedding_model: str) -> FAISS:
    embeddings = build_embeddings(embedding_model)
    index_file = vectorstore_dir / "index.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            f"Vector store not found at {vectorstore_dir}."
        )
    return FAISS.load_local(
        str(vectorstore_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
