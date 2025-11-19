"""App-level configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


@dataclass(slots=True)
class AppSettings:
    docs_dir: Path
    vectorstore_dir: Path
    embedding_model: str
    waypoint_docs_dir: Path
    llm_provider: str
    llm_model: str
    temperature: float


def load_settings() -> AppSettings:
    root = Path(__file__).resolve().parents[4]
    return AppSettings(
        docs_dir=Path(os.getenv("DOCS_DIR", root / "data" / "seed")).resolve(),
        vectorstore_dir=Path(os.getenv("VECTORSTORE_DIR", root / "artifacts" / "vectorstore")).resolve(),
        embedding_model=os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        waypoint_docs_dir=Path(os.getenv("WAYPOINT_DIR", root / "data" / "seed")).resolve(),
        llm_provider=os.getenv("LLM_PROVIDER", "langgraph"),
        llm_model=os.getenv("LLM_MODEL", "llama-3.1-8b-instruct"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.2")),
    )
