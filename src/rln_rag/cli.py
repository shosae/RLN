from __future__ import annotations

from pathlib import Path
from typing import Optional
import json

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .llm import build_llm
from .plan_executor_graph import build_plan_executor_graph
from .planner_graph import generate_plan
from .plan_validator import validate_plan
from .orchestrator_graph import build_orchestrator_graph
from .settings import Settings, get_settings
from .vectorstore import build_vectorstore, load_vectorstore

console = Console()
app = typer.Typer(
    name="rln-rag",
    help="LangGraph-based RAG playground for the RLN experiments.",
)


def _settings_table(settings: Settings) -> Table:
    table = Table(box=box.MINIMAL_DOUBLE_HEAD)
    table.add_column("Setting", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")
    table.add_row("Docs dir", str(settings.docs_dir))
    table.add_row("Vector store dir", str(settings.vectorstore_dir))
    table.add_row("Embeddings", settings.embedding_model)
    table.add_row("Retriever k", str(settings.top_k))
    table.add_row("Chunk size/overlap", f"{settings.chunk_size}/{settings.chunk_overlap}")
    table.add_row("LLM provider", settings.llm_provider)
    table.add_row("LLM model", settings.llm_model)
    table.add_row("LangGraph base URL", settings.langgraph_base_url)
    return table


@app.command()
def info() -> None:
    """Print the current runtime configuration."""
    settings = get_settings()
    console.print(_settings_table(settings))
    missing = []
    if settings.llm_provider.lower() == "langgraph" and not settings.langgraph_api_key:
        missing.append("LANGGRAPH_API_KEY")
    if settings.llm_provider.lower() == "groq" and not settings.groq_api_key:
        missing.append("GROQ_API_KEY")
    if missing:
        console.print(
            Panel(
                "[yellow]Set the following env vars before running `ask`: "
                + ", ".join(missing),
                title="Missing credentials",
                expand=False,
            )
        )


@app.command()
def ingest(force: bool = typer.Option(False, "--force", "-f", help="Rebuild the FAISS store even if it exists.")) -> None:
    """
    Load local documents, split them into chunks, and persist the FAISS vector store.
    """
    settings = get_settings()
    if not force:
        index_file = Path(settings.vectorstore_dir) / "index.faiss"
        if index_file.exists():
            console.print(
                Panel(
                    f"[green]Vector store already present at {settings.vectorstore_dir}. "
                    "Use --force to rebuild.",
                    title="Ingest skipped",
                    expand=False,
                )
            )
            raise typer.Exit()
    console.print(f"[bold]Building vector store from {settings.docs_dir}...")
    store = build_vectorstore(settings)
    console.print(
        Panel(
            f"[green]Stored {store.index.ntotal} chunks at {settings.vectorstore_dir}",
            title="Ingest complete",
            expand=False,
        )
    )


@app.command()
def ask(
    question: str = typer.Argument(..., help="Natural language question to ask the RLN knowledge base."),
    top_k: Optional[int] = typer.Option(None, "--top-k", "-k", help="Override the number of chunks retrieved."),
) -> None:
    """
    Run the LangGraph RAG pipeline for a single question.
    """
    settings = get_settings()
    if top_k is not None:
        settings.top_k = top_k

    try:
        vectorstore = load_vectorstore(settings)
    except FileNotFoundError as exc:
        console.print(
            Panel(
                "[red]Vector store missing. Run `rln-rag ingest` first.",
                title="No vector store",
                expand=False,
            )
        )
        raise typer.Exit(code=1) from exc

    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.top_k})
    llm = build_llm(settings)
    graph = build_orchestrator_graph(retriever, llm)

    console.print(f"[bold]Question:[/bold] {question}")
    state = graph.invoke({"question": question})
    answer = state.get("answer", "").strip()
    console.print(Panel(answer or "No answer generated.", title=settings.llm_model, expand=False))

    plan_obj = state.get("plan")
    if plan_obj:
        console.print(Panel(json.dumps(plan_obj, ensure_ascii=False, indent=2), title="PLAN", expand=False))
        validation = state.get("plan_validation")
        if validation and getattr(validation, "errors", None):
            error_text = "\n".join(f"- {msg}" for msg in validation.errors)
            console.print(Panel(error_text, title="PLAN validation errors", style="red", expand=False))
        if validation and getattr(validation, "warnings", None):
            warn_text = "\n".join(f"- {msg}" for msg in validation.warnings)
            console.print(Panel(warn_text, title="PLAN validation warnings", style="yellow", expand=False))

    context_docs = state.get("context") or []
    if not context_docs:
        console.print("[yellow]No documents were retrieved.")
        return

    table = Table(title="Retrieved context", show_lines=False, box=box.SIMPLE_HEAVY)
    table.add_column("#", style="cyan", no_wrap=True)
    table.add_column("Source", style="green", no_wrap=True)
    table.add_column("Preview", style="white")
    for idx, doc in enumerate(context_docs, start=1):
        meta = doc.metadata or {}
        source = meta.get("source") or meta.get("file_path") or meta.get("title") or "snippet"
        snippet = " ".join(doc.page_content.split())
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        table.add_row(str(idx), str(source), snippet)
    console.print(table)


@app.command()
def plan(
    question: str = typer.Argument(..., help="Natural language instruction to convert into a PLAN JSON."),
    top_k: Optional[int] = typer.Option(None, "--top-k", "-k", help="Override retriever depth for planning."),
) -> None:
    """
    Generate a PLAN JSON using the planner LLM with enforced JSON-only rules.
    """
    settings = get_settings()
    if top_k is not None:
        settings.top_k = top_k

    try:
        vectorstore = load_vectorstore(settings)
    except FileNotFoundError as exc:
        console.print(
            Panel(
                "[red]Vector store missing. Run `rln-rag ingest` first.",
                title="No vector store",
                expand=False,
            )
        )
        raise typer.Exit(code=1) from exc

    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.top_k})
    llm = build_llm(settings)

    plan_obj, docs = generate_plan(question, retriever, llm)
    console.print(Panel(json.dumps(plan_obj, ensure_ascii=False, indent=2), title="PLAN"))

    validation = validate_plan(plan_obj)
    if validation.errors:
        error_text = "\n".join(f"- {msg}" for msg in validation.errors)
        console.print(Panel(error_text, title="PLAN validation errors", style="red", expand=False))
    if validation.warnings:
        warn_text = "\n".join(f"- {msg}" for msg in validation.warnings)
        console.print(Panel(warn_text, title="PLAN validation warnings", style="yellow", expand=False))

    if not docs:
        console.print("[yellow]No documents were retrieved for context.")
        return

    table = Table(title="Retrieved context for planning", show_lines=False, box=box.SIMPLE_HEAVY)
    table.add_column("#", style="cyan", no_wrap=True)
    table.add_column("Source", style="green", no_wrap=True)
    table.add_column("Preview", style="white")
    for idx, doc in enumerate(docs, start=1):
        meta = doc.metadata or {}
        source = meta.get("source") or meta.get("file_path") or meta.get("title") or "snippet"
        snippet = " ".join(doc.page_content.split())
        if len(snippet) > 160:
            snippet = snippet[:157] + "..."
        table.add_row(str(idx), str(source), snippet)
    console.print(table)


@app.command("run-plan")
def run_plan(
    instruction: str = typer.Argument(..., help="Natural language mission to plan and execute."),
    top_k: Optional[int] = typer.Option(None, "--top-k", "-k", help="Override retriever depth for planning."),
) -> None:
    """
    Plan-and-Execute pipeline. Generates a PLAN and executes each action sequentially.
    """
    settings = get_settings()
    if top_k is not None:
        settings.top_k = top_k

    try:
        vectorstore = load_vectorstore(settings)
    except FileNotFoundError as exc:
        console.print(
            Panel(
                "[red]Vector store missing. Run `rln-rag ingest` first.",
                title="No vector store",
                expand=False,
            )
        )
        raise typer.Exit(code=1) from exc

    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.top_k})
    llm = build_llm(settings)
    graph = build_plan_executor_graph(retriever, llm)

    console.print(f"[bold]Instruction:[/bold] {instruction}")
    state = graph.invoke({"input": instruction})
    response = state.get("response") or "No response."
    console.print(Panel(response, title="Plan-and-Execute", expand=False))

    errors = state.get("validation_errors") or []
    if errors:
        console.print(
            Panel(
                "\n".join(f"- {msg}" for msg in errors),
                title="PLAN validation errors",
                style="red",
                expand=False,
            )
        )

    past_steps = state.get("past_steps") or []
    if past_steps:
        table = Table(title="Executed Steps", show_lines=False, box=box.SIMPLE_HEAVY)
        table.add_column("#", style="cyan", no_wrap=True)
        table.add_column("Action", style="green", no_wrap=True)
        table.add_column("Result", style="white")
        for idx, (step, result) in enumerate(past_steps, start=1):
            action = step.get("action")
            result_text = result.get("message") or result.get("status") or str(result)
            table.add_row(str(idx), action or "-", result_text)
        console.print(table)


if __name__ == "__main__":
    app()
