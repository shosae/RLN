# RLN RAG Playground

LangGraph-based Retrieval Augmented Generation experiments targeting Llama 3.1 8B Instruct.

## Prerequisites

- Python 3.10+
- Local CPU is sufficient (FAISS + sentence-transformers). A LangGraph-hosted Llama 3.1 8B endpoint is used for generation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Configure credentials (the LangGraph key is mandatory when `LLM_PROVIDER=langgraph`):

```bash
export LANGGRAPH_API_KEY="lg_apikey"
# Optional overrides:
# export DOCS_DIR="data/seed"
# export VECTORSTORE_DIR="artifacts/vectorstore"
# export LLM_MODEL="llama-3.1-8b-instruct"
```

## CLI Usage

1. Inspect configuration:

   ```bash
   rln-rag info
   ```

2. Build the FAISS vector store from `data/seed`:

   ```bash
   rln-rag ingest
   # use --force to rebuild
   ```

3. Query through the LangGraph pipeline:

   ```bash
   rln-rag ask "What is RLN?"
   ```

The `ask` command prints the generated answer followed by the retrieved context snippets with their source labels so you can verify grounding.

## Environment Variables

| Name | Default | Notes |
| --- | --- | --- |
| `DOCS_DIR` | `data/seed` | Directory containing `.md`/`.txt` docs to ingest. |
| `VECTORSTORE_DIR` | `artifacts/vectorstore` | FAISS index output. |
| `LLM_PROVIDER` | `langgraph` | One of `langgraph`, `groq`, `ollama`. |
| `LLM_MODEL` | `llama-3.1-8b-instruct` | Forwarded to the configured provider. |
| `LANGGRAPH_BASE_URL` | `https://api.langgraph.com/v1` | Override when self-hosting. |
| `LANGGRAPH_API_KEY` | _required_ | Needed when using LangGraph (preferred). |
| `GROQ_API_KEY` |  | Used when `LLM_PROVIDER=groq`. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Used when `LLM_PROVIDER=ollama`. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `700` / `150` | Controls splitter behavior. |
| `TOP_K` | `4` | Retriever depth. |
