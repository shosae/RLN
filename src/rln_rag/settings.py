from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _guess_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


PROJECT_ROOT = _guess_project_root()


def _path_from_env(name: str, default: str) -> Path:
    value = os.getenv(name)
    if value:
        return Path(value).expanduser().resolve()

    default_path = Path(default)
    if not default_path.is_absolute():
        default_path = PROJECT_ROOT / default_path
    return default_path.expanduser().resolve()


@dataclass(slots=True)
class Settings:
    """Runtime configuration sourced from environment variables."""

    docs_dir: Path = _path_from_env("DOCS_DIR", "data/seed")
    vectorstore_dir: Path = _path_from_env("VECTORSTORE_DIR", "artifacts/vectorstore")
    embedding_model: str = os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    llm_provider: str = os.getenv("LLM_PROVIDER", "langgraph")
    llm_model: str = os.getenv("LLM_MODEL", "llama-3.1-8b-instruct")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "700"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))
    langgraph_api_key: str | None = os.getenv("LANGGRAPH_API_KEY")
    langgraph_base_url: str = os.getenv("LANGGRAPH_BASE_URL", "https://api.langgraph.com/v1")
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def ensure_directories(self) -> None:
        self.docs_dir.mkdir(parents=True, exist_ok=True)
        self.vectorstore_dir.mkdir(parents=True, exist_ok=True)


def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings


settings = get_settings()
