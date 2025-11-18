from __future__ import annotations

from pathlib import Path
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .settings import Settings


def _loader(path: Path, pattern: str) -> List[Document]:
    loader = DirectoryLoader(
        str(path),
        glob=pattern,
        show_progress=True,
        use_multithreading=True,
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    return loader.load()


def load_documents(settings: Settings) -> List[Document]:
    docs: List[Document] = []
    docs.extend(_loader(settings.docs_dir, "**/*.md"))
    docs.extend(_loader(settings.docs_dir, "**/*.txt"))
    docs.extend(_loader(settings.docs_dir, "**/*.jsonl"))
    return docs


def split_documents(documents: List[Document], settings: Settings) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    return splitter.split_documents(documents)


def build_embeddings(settings: Settings) -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def build_vectorstore(settings: Settings) -> FAISS:
    documents = split_documents(load_documents(settings), settings)
    embeddings = build_embeddings(settings)
    store = FAISS.from_documents(documents, embeddings)
    settings.vectorstore_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(settings.vectorstore_dir))
    return store


def load_vectorstore(settings: Settings) -> FAISS:
    embeddings = build_embeddings(settings)
    index_path = settings.vectorstore_dir
    index_file = index_path / "index.faiss"
    if not index_file.exists():
        raise FileNotFoundError(
            f"Vector store not found at {index_path}. Run `rln-rag ingest` first."
        )
    return FAISS.load_local(
        str(index_path),
        embeddings,
        allow_dangerous_deserialization=True,
    )
