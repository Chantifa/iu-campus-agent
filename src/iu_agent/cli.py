"""Command line interface: interactive chat (REPL), ingestion, search and myCampus commands."""

from __future__ import annotations

import datetime as dt
import shlex
import uuid
import webbrowser
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import typer
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from rich.panel import Panel
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from iu_agent import __version__
from iu_agent.agent.graph import NODE_AGENT, NODE_TOOLS, build_graph
from iu_agent.agent.prompts import build_system_prompt
from iu_agent.agent.tools import AgentContext, build_tools
from iu_agent.config import Settings, load_settings
from iu_agent.models import (
    PROVIDER_ANTHROPIC,
    PROVIDER_HINTS,
    PROVIDER_LABELS,
    ModelRef,
    ProviderNotConfigured,
    build_chat_model,
    configured_providers,
    default_model_ref,
    list_models,
    parse_model_ref,
)
from iu_agent.moodle.auth import (
    clear_token,
    launch_url,
    load_token,
    make_passport,
    parse_launch_response,
    save_token,
    token_info,
)
from iu_agent.moodle.client import MoodleClient, MoodleError
from iu_agent.rag.ingest import Ingestor, Manifest
from iu_agent.rag.store import CourseVectorStore
from iu_agent.ui import (
    HELP_TEXT,
    StreamRenderer,
    ask_approval,
    banner,
    interactive_terminal,
    make_console,
    models_table,
    render_tool_call,
    render_tool_result,
)

app = typer.Typer(
    help="IU Campus Agent: a Claude-Code-style CLI agent with a RAG over your IU course material.",
    add_completion=False,
    rich_markup_mode="rich",
    invoke_without_command=True,
    no_args_is_help=False,
)
moodle_app = typer.Typer(help="myCampus (Moodle) login, status and content sync.")
app.add_typer(moodle_app, name="moodle")

console = make_console()

SLASH_COMMANDS = {
    "/help": "show help",
    "/model": "choose or switch the model",
    "/models": "list models ('/models live' queries the APIs)",
    "/search": "raw retrieval without the LLM",
    "/courses": "indexed courses",
    "/ingest": "index the documents folder",
    "/moodle": "moodle sync | status",
    "/clear": "new conversation",
    "/status": "configuration and index statistics",
    "/exit": "quit",
}


# ============================================================================= shared helpers
def _open_store(settings: Settings) -> CourseVectorStore:
    return CourseVectorStore(settings)


def _moodle_client(settings: Settings) -> MoodleClient | None:
    token = load_token(settings)
    return MoodleClient(settings.moodle_url, token) if token else None


def _progress() -> Progress:
    return Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=True,
    )


def _run_ingest(settings: Settings, store: CourseVectorStore, path: Path | None, *, force: bool, prune: bool):
    ingestor = Ingestor(settings, store)
    documents = ingestor.collect_folder(path)
    console.print(f"Found {len(documents)} supported files under {path or settings.iu_docs_path}")
    with _progress() as progress:
        task = progress.add_task("indexing", total=len(documents))

        def on_progress(done: int, total: int, name: str) -> None:
            progress.update(task, completed=done, description=f"indexing {name[:40]}")

        report = ingestor.ingest(documents, force=force, on_progress=on_progress)
    if prune:
        report.removed = ingestor.prune("onedrive", {d.source for d in documents})
    console.print(f"[green]Ingestion finished:[/green] {report.summary()}")
    for source, error in report.failed[:25]:
        console.print(f"  [red]failed[/red] {source}: {error}")
    if len(report.failed) > 25:
        console.print(f"  ... {len(report.failed) - 25} more failures")
    return report


def _chunk_text(chunk: Any) -> str:
    content = getattr(chunk, "content", "")
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


class SlashCompleter(Completer):
    def get_completions(self, document, complete_event) -> Iterable[Completion]:
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        for command, description in SLASH_COMMANDS.items():
            if command.startswith(text):
                yield Completion(command, start_position=-len(text), display_meta=description)


# ============================================================================= chat session
class ChatSession:
    def __init__(self, settings: Settings, *, approval_mode: str = "ask", no_rag: bool = False) -> None:
        self.settings = settings
        self.manifest = Manifest(settings.manifest_path)
        self._store: CourseVectorStore | None = None
        self._store_failed = False
        self.no_rag = no_rag
        self.workspace = Path(settings.workspace_dir).resolve()
        self.moodle = _moodle_client(settings)
        self.ctx = AgentContext(
            settings=settings,
            manifest=self.manifest,
            store_factory=self.store,
            workspace=self.workspace,
            moodle=self.moodle,
            approval_mode=approval_mode,
        )
        self.tools = build_tools(self.ctx)
        self.checkpointer = InMemorySaver()
        self.thread_id = str(uuid.uuid4())
        self.model_ref: ModelRef | None = None
        self.llm = None
        self.graph = None
        self._system_prompt: str | None = None

    # ------------------------------------------------------------------ infrastructure
    def store(self) -> CourseVectorStore | None:
        if self.no_rag or self._store_failed:
            return None
        if self._store is None:
            try:
                self._store = _open_store(self.settings)
            except Exception as exc:
                self._store_failed = True
                console.print(f"[red]Vector store unavailable:[/red] {exc}")
                return None
        return self._store

    @property
    def config(self) -> dict:
        return {"configurable": {"thread_id": self.thread_id}}

    def system_prompt(self) -> str:
        if self._system_prompt is None:
            self._system_prompt = build_system_prompt(
                courses=self.manifest.courses(),
                total_documents=len(self.manifest.documents),
                total_chunks=self.manifest.total_chunks(),
                workspace=self.workspace,
                moodle_connected=self.moodle is not None,
                today=dt.date.today().isoformat(),
            )
        return self._system_prompt

    def invalidate_prompt(self) -> None:
        self._system_prompt = None

    def set_model(self, ref: ModelRef) -> None:
        self.llm = build_chat_model(ref, self.settings)
        self.model_ref = ref
        self.graph = build_graph(
            self.llm,
            self.tools,
            self.system_prompt,
            self.checkpointer,
            context_budget_tokens=self.settings.context_budget_tokens,
            cache_system_prompt=ref.provider == PROVIDER_ANTHROPIC,
        )

    # ------------------------------------------------------------------ model selection
    def choose_model(
        self, requested: str | None = None, *, interactive: bool = True, force_prompt: bool = False
    ) -> bool:
        if requested:
            try:
                self.set_model(parse_model_ref(requested, self.settings))
            except (ValueError, ProviderNotConfigured) as exc:
                console.print(f"[red]{exc}[/red]")
                return False
            return True
        interactive = interactive and interactive_terminal()
        if not force_prompt and self.settings.default_model:
            return self.choose_model(self.settings.default_model, interactive=interactive)
        providers = configured_providers(self.settings)
        if not providers:
            console.print(
                Panel(
                    "No LLM provider is configured, so only /search and /courses work.\n"
                    + "\n".join(f"- {PROVIDER_LABELS[p]}: {hint}" for p, hint in PROVIDER_HINTS.items())
                    + "\n\nPut the keys into a .env file (see .env.example) and restart.",
                    title="[yellow]No model available[/yellow]",
                    border_style="yellow",
                )
            )
            return False
        if not interactive:
            ref = default_model_ref(self.settings)
            return self.choose_model(ref.id, interactive=False) if ref else False
        models, warnings = list_models(self.settings)
        for warning in warnings:
            console.print(f"[yellow]{warning}[/yellow]")
        console.print(models_table(models, self.model_ref))
        default = self.model_ref or default_model_ref(self.settings) or models[0].ref
        from prompt_toolkit.shortcuts import prompt

        while True:
            try:
                answer = prompt(f"Which model? [number or id, Enter = {default.id}] ").strip()
            except (KeyboardInterrupt, EOFError):
                answer = ""
            if not answer:
                chosen = default
            elif answer.isdigit() and 1 <= int(answer) <= len(models):
                chosen = models[int(answer) - 1].ref
            else:
                try:
                    chosen = parse_model_ref(answer, self.settings)
                except ValueError as exc:
                    console.print(f"[red]{exc}[/red]")
                    continue
            try:
                self.set_model(chosen)
            except ProviderNotConfigured as exc:
                console.print(f"[red]{exc}[/red]")
                continue
            console.print(f"Using [bold green]{chosen.id}[/bold green]")
            return True

    # ------------------------------------------------------------------ conversation
    def _repair_dangling_tool_calls(self) -> None:
        """After Ctrl+C mid-turn the last message may be a tool call without result; close it."""
        if self.graph is None:
            return
        try:
            state = self.graph.get_state(self.config)
        except Exception:
            return
        messages = state.values.get("messages") if state and state.values else None
        if not messages:
            return
        last = messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            fillers = [
                ToolMessage(content="Cancelled by the user before the tool ran.", tool_call_id=call["id"])
                for call in last.tool_calls
            ]
            self.graph.update_state(self.config, {"messages": fillers}, as_node=NODE_TOOLS)

    def run_turn(self, user_text: str) -> str:
        if self.graph is None:
            console.print("[yellow]No model selected. Use /model first.[/yellow]")
            return ""
        payload: Any = {"messages": [HumanMessage(content=user_text)]}
        final_text = ""
        while True:
            renderer = StreamRenderer(console)
            renderer.thinking()
            interrupts: list[Any] = []
            try:
                for mode, data in self.graph.stream(
                    payload, self.config, stream_mode=["messages", "updates"]
                ):
                    if mode == "messages":
                        chunk, metadata = data
                        if metadata.get("langgraph_node") == NODE_AGENT and isinstance(chunk, AIMessageChunk):
                            renderer.append(_chunk_text(chunk))
                    elif mode == "updates":
                        for node, update in data.items():
                            if node == "__interrupt__":
                                interrupts.extend(update)
                            elif node == NODE_AGENT:
                                message = update["messages"][-1]
                                if isinstance(message, AIMessage) and not renderer.buffer:
                                    # the provider did not stream tokens: show the whole answer now
                                    renderer.append(_chunk_text(message))
                                if isinstance(message, AIMessage) and message.tool_calls:
                                    renderer.finish()
                                    for call in message.tool_calls:
                                        render_tool_call(console, call["name"], call.get("args") or {})
                            elif node == NODE_TOOLS:
                                for message in update.get("messages", []):
                                    if isinstance(message, ToolMessage):
                                        render_tool_result(console, message.name or "tool", message.content)
                                renderer.thinking()
            except KeyboardInterrupt:
                renderer.finish()
                console.print("[yellow]Interrupted.[/yellow]")
                self._repair_dangling_tool_calls()
                return final_text
            except Exception as exc:
                renderer.finish()
                console.print(f"[red]{type(exc).__name__}:[/red] {exc}")
                self._repair_dangling_tool_calls()
                return final_text
            final_text = renderer.finish() or final_text
            if not interrupts:
                return final_text
            resume: dict[str, Any] = {}
            for request in interrupts:
                decision = ask_approval(
                    console, request.value if isinstance(request.value, dict) else {"value": request.value}
                )
                interrupt_id = getattr(request, "id", None)
                if interrupt_id:
                    resume[interrupt_id] = decision
                else:
                    payload = Command(resume=decision)
                    break
            else:
                payload = Command(resume=resume)

    # ------------------------------------------------------------------ slash commands
    def handle_command(self, line: str) -> bool:
        """Return False when the session should end."""
        parts = line.strip().split(maxsplit=1)
        command = parts[0].lower()
        argument = parts[1].strip() if len(parts) > 1 else ""
        if command in ("/exit", "/quit", "/q"):
            return False
        if command == "/help":
            console.print(HELP_TEXT)
        elif command == "/model":
            self.choose_model(argument or None, force_prompt=True)
        elif command == "/models":
            models, warnings = list_models(self.settings, live=argument.lower() == "live")
            for warning in warnings:
                console.print(f"[yellow]{warning}[/yellow]")
            console.print(models_table(models, self.model_ref))
        elif command == "/search":
            if not argument:
                console.print("Usage: /search <query>")
            else:
                self.raw_search(argument)
        elif command == "/courses":
            self.print_courses()
        elif command == "/ingest":
            store = self.store()
            if store is None:
                console.print("[red]Vector store unavailable.[/red]")
            else:
                path = Path(argument).expanduser() if argument else None
                _run_ingest(self.settings, store, path, force=False, prune=path is None)
                self.manifest = Manifest(self.settings.manifest_path)
                self.ctx.manifest = self.manifest
                self.invalidate_prompt()
        elif command == "/moodle":
            self.moodle_command(argument)
        elif command == "/clear":
            self.thread_id = str(uuid.uuid4())
            console.print("[dim]New conversation started.[/dim]")
        elif command == "/status":
            print_status(self.settings, self.manifest, self.model_ref)
        else:
            console.print(f"Unknown command {command}. /help lists the commands.")
        return True

    def raw_search(self, query: str, k: int | None = None, course: str | None = None) -> None:
        store = self.store()
        if store is None:
            console.print("[red]Vector store unavailable.[/red]")
            return
        hits = store.search(query, k=k or self.settings.retrieval_k, course=course)
        print_hits(hits)

    def print_courses(self) -> None:
        courses = self.manifest.courses()
        if not courses:
            console.print("Nothing indexed yet. Use /ingest or `iu-agent moodle sync`.")
            return
        table = Table(title="Indexed courses")
        table.add_column("Course")
        table.add_column("Documents", justify="right")
        for name, count in courses.items():
            table.add_row(name, str(count))
        console.print(table)

    def moodle_command(self, argument: str) -> None:
        action = (argument.split() or ["status"])[0].lower()
        if action == "status":
            print_moodle_status(self.settings)
        elif action == "sync":
            if self.moodle is None:
                console.print("Not logged in. Run `iu-agent moodle login` in a terminal first.")
                return
            store = self.store()
            if store is None:
                console.print("[red]Vector store unavailable.[/red]")
                return
            run_moodle_sync(self.settings, store, self.moodle, course_ids=None, force=False)
            self.manifest = Manifest(self.settings.manifest_path)
            self.ctx.manifest = self.manifest
            self.invalidate_prompt()
        else:
            console.print("Usage: /moodle status | /moodle sync")

    # ------------------------------------------------------------------ main loop
    def loop(self) -> None:
        bindings = KeyBindings()

        @bindings.add("escape", "enter")
        def _newline(event) -> None:
            event.current_buffer.insert_text("\n")

        session: PromptSession = PromptSession(
            history=FileHistory(str(self.settings.history_path)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=SlashCompleter(),
            complete_while_typing=True,
            key_bindings=bindings,
            bottom_toolbar=self._toolbar,
        )
        while True:
            try:
                line = session.prompt(HTML("<b><ansicyan>❯</ansicyan></b> "))
            except KeyboardInterrupt:
                continue
            except EOFError:
                break
            text = line.strip()
            if not text:
                continue
            if text.startswith("/"):
                if not self.handle_command(text):
                    break
                continue
            self.run_turn(text)
        console.print("[dim]Bye.[/dim]")

    def _toolbar(self) -> HTML:
        model = self.model_ref.id if self.model_ref else "no model"
        chunks = self.manifest.total_chunks()
        return HTML(
            f" <b>{model}</b> | index: {len(self.manifest.documents)} docs / {chunks} chunks"
            f" | thread {self.thread_id[:8]} | /help"
        )

    def show_banner(self) -> None:
        store_location = (
            "disabled"
            if self.no_rag
            else (self.settings.qdrant_url or f"embedded ({self.settings.qdrant_local_path})")
        )
        info = token_info(self.settings)
        moodle = (
            f"connected as {info.get('fullname') or info.get('username')}"
            if self.moodle and info
            else ("token from environment" if self.moodle else "not connected (iu-agent moodle login)")
        )
        banner(
            console,
            version=__version__,
            model=self.model_ref,
            store_location=store_location,
            documents=len(self.manifest.documents),
            chunks=self.manifest.total_chunks(),
            courses=len(self.manifest.courses()),
            moodle=moodle,
            workspace=str(self.workspace),
        )


# ============================================================================= printing helpers
def print_hits(hits: list[tuple[Any, float]]) -> None:
    if not hits:
        console.print("No results.")
        return
    for index, (doc, score) in enumerate(hits, start=1):
        meta = doc.metadata or {}
        where = " | ".join(str(meta[key]) for key in ("course", "file_name", "module") if meta.get(key))
        page = (
            f" p.{meta['page']}"
            if meta.get("page")
            else (f" slide {meta['slide']}" if meta.get("slide") else "")
        )
        body = (
            doc.page_content.split("\n", 1)[1]
            if doc.page_content.startswith("[") and "\n" in doc.page_content
            else doc.page_content
        )
        console.print(
            Panel(
                body[:1200],
                title=f"[bold]{index}. {where}{page}[/bold]  score {score:.3f}",
                border_style="blue",
            )
        )


def print_status(settings: Settings, manifest: Manifest, model: ModelRef | None = None) -> None:
    table = Table(title="IU Campus Agent status", show_header=False)
    table.add_column("Key", style="bold")
    table.add_column("Value")
    providers = configured_providers(settings)
    table.add_row("Version", __version__)
    table.add_row("Model", model.id if model else (settings.default_model or "(asked at start)"))
    table.add_row("Providers", ", ".join(PROVIDER_LABELS[p] for p in providers) or "none configured")
    table.add_row("Vector DB", settings.qdrant_url or f"embedded local mode ({settings.qdrant_local_path})")
    table.add_row(
        "Embeddings",
        f"{settings.embedding_model} (+ {settings.sparse_model} sparse)"
        if settings.hybrid_search
        else settings.embedding_model,
    )
    table.add_row("Documents folder", str(settings.iu_docs_path))
    table.add_row("Excluded", settings.iu_exclude_globs or "-")
    table.add_row(
        "Indexed",
        f"{len(manifest.documents)} documents, {manifest.total_chunks()} chunks, "
        f"{len(manifest.courses())} courses",
    )
    by_origin: dict[str, int] = {}
    for entry in manifest.documents.values():
        by_origin[entry.get("origin", "?")] = by_origin.get(entry.get("origin", "?"), 0) + 1
    table.add_row("Sources", ", ".join(f"{k}: {v}" for k, v in by_origin.items()) or "-")
    table.add_row(
        "myCampus", settings.moodle_url + (" (token stored)" if load_token(settings) else " (not logged in)")
    )
    table.add_row("Data dir", str(settings.data_dir.resolve()))
    table.add_row("Workspace", str(Path(settings.workspace_dir).resolve()))
    console.print(table)


def print_moodle_status(settings: Settings) -> None:
    info = token_info(settings)
    if not load_token(settings):
        console.print("Not logged in. Run `iu-agent moodle login`.")
        return
    if info:
        console.print(
            f"Logged in to {info.get('sitename')} as {info.get('fullname')} "
            f"({info.get('username')}), user id {info.get('userid')}"
        )
    else:
        console.print("Token provided through MOODLE_TOKEN.")


def run_moodle_sync(
    settings: Settings,
    store: CourseVectorStore,
    client: MoodleClient,
    *,
    course_ids: list[int] | None,
    force: bool,
) -> None:
    from iu_agent.moodle.sync import MoodleSync

    ingestor = Ingestor(settings, store)
    sync = MoodleSync(settings, client, ingestor, log=lambda msg: console.print(f"[dim]{msg}[/dim]"))
    with _progress() as progress:
        task = progress.add_task("syncing myCampus", total=None)

        def on_progress(done: int, total: int, name: str) -> None:
            progress.update(task, total=total, completed=done, description=f"indexing {name[:40]}")

        report, plan = sync.run(course_ids, force=force, on_progress=on_progress)
    console.print(f"[green]myCampus sync finished:[/green] {report.summary()}")
    for warning in plan.warnings[:20]:
        console.print(f"  [yellow]warning[/yellow] {warning}")
    if plan.skipped:
        console.print(
            f"  [dim]{len(plan.skipped)} activities skipped "
            "(videos, quizzes, SCORM, external tools ...)[/dim]"
        )
    for source, error in report.failed[:20]:
        console.print(f"  [red]failed[/red] {source}: {error}")


# ============================================================================= commands
@app.callback()
def main(
    ctx: typer.Context,
    model: str = typer.Option(
        None, "--model", "-m", help="Model to use, e.g. anthropic:claude-opus-5 or kimi:kimi-k3."
    ),
    version: bool = typer.Option(False, "--version", help="Print the version and exit."),
) -> None:
    if version:
        console.print(f"iu-agent {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        _chat(model=model, no_rag=False)


@app.command()
def chat(
    model: str = typer.Option(
        None, "--model", "-m", help="Model to use, e.g. anthropic:claude-opus-5 or kimi:kimi-k3."
    ),
    no_rag: bool = typer.Option(False, "--no-rag", help="Do not open the vector store."),
) -> None:
    """Start the interactive agent (default command)."""
    _chat(model=model, no_rag=no_rag)


def _chat(model: str | None, no_rag: bool) -> None:
    settings = load_settings()
    session = ChatSession(settings, no_rag=no_rag)
    session.choose_model(model)
    session.show_banner()
    session.loop()


@app.command()
def ask(
    question: str = typer.Argument(..., help="The question or task."),
    model: str = typer.Option(None, "--model", "-m", help="Model to use."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Auto-approve file writes and shell commands."),
) -> None:
    """Ask a single question non-interactively and print the answer."""
    settings = load_settings()
    session = ChatSession(settings, approval_mode="allow" if yes else "deny")
    if not session.choose_model(model, interactive=False):
        raise typer.Exit(code=1)
    session.run_turn(question)


@app.command()
def ingest(
    path: Path = typer.Argument(None, help="Folder to index (default: IU_DOCS_PATH)."),
    force: bool = typer.Option(False, "--force", help="Re-embed every file even if unchanged."),
    reset: bool = typer.Option(False, "--reset", help="Drop the collection and the manifest first."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only list the files that would be indexed."),
    no_prune: bool = typer.Option(False, "--no-prune", help="Keep index entries of files that disappeared."),
) -> None:
    """Index the OneDrive/IU folder (or another folder) into the vector store."""
    settings = load_settings()
    if dry_run:
        ingestor = Ingestor.__new__(Ingestor)
        ingestor.settings = settings
        documents = Ingestor.collect_folder(ingestor, path)
        for document in documents:
            console.print(f"{document.meta.get('course'):40} {document.meta.get('rel_path')}")
        console.print(f"{len(documents)} files would be indexed.")
        return
    store = _open_store(settings)
    try:
        if reset:
            Ingestor(settings, store).remove_all()
            console.print("[yellow]Collection and manifest reset.[/yellow]")
        _run_ingest(settings, store, path, force=force, prune=not no_prune and path is None)
    finally:
        store.close()


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query."),
    k: int = typer.Option(6, "--k", help="Number of results."),
    course: str = typer.Option(None, "--course", help="Restrict to one course."),
) -> None:
    """Search the vector store directly (no LLM)."""
    settings = load_settings()
    store = _open_store(settings)
    try:
        from iu_agent.agent.tools import resolve_course

        resolved = resolve_course(course, Manifest(settings.manifest_path)) if course else None
        print_hits(store.search(query, k=k, course=resolved))
    finally:
        store.close()


@app.command()
def models(
    live: bool = typer.Option(False, "--live", help="Query the provider APIs for their model lists."),
) -> None:
    """List the models that can be selected."""
    settings = load_settings()
    found, warnings = list_models(settings, live=live)
    for warning in warnings:
        console.print(f"[yellow]{warning}[/yellow]")
    if not found:
        console.print(
            "No provider configured. "
            + "; ".join(f"{PROVIDER_LABELS[p]}: {h}" for p, h in PROVIDER_HINTS.items())
        )
        return
    console.print(models_table(found, default_model_ref(settings)))


@app.command()
def status() -> None:
    """Show configuration, providers and index statistics."""
    settings = load_settings()
    print_status(settings, Manifest(settings.manifest_path))


# ----------------------------------------------------------------------------- moodle
@moodle_app.command("login")
def moodle_login(
    token: str = typer.Option(
        None, "--token", help="Paste a token or a moodlemobile://token=... URL directly."
    ),
    no_browser: bool = typer.Option(
        False, "--no-browser", help="Only print the login URL instead of opening it."
    ),
) -> None:
    """Log in to myCampus through the browser (SSO) and store the web-service token."""
    settings = load_settings()
    passport: str | None = None
    if not token:
        try:
            config = MoodleClient.public_config(settings.moodle_url)
        except Exception as exc:
            console.print(f"[red]Cannot read the public site configuration:[/red] {exc}")
            raise typer.Exit(code=1) from exc
        if not config.get("enablemobilewebservice"):
            console.print(
                "[red]The mobile web service is disabled on this site; "
                "ask the administrators for a token.[/red]"
            )
            raise typer.Exit(code=1)
        passport = make_passport()
        url = launch_url(settings.moodle_url, passport, settings.moodle_service, config.get("launchurl"))
        console.print(
            Panel(
                f"1. A browser window opens the myCampus login ({config.get('sitename')}).\n"
                "2. Log in with your IU account (SSO).\n"
                "3. The browser is then redirected to an address starting with [bold]moodlemobile://token=[/bold].\n"
                "   It cannot open that address, but shows it in the address bar or on the error page.\n"
                "4. Copy the whole address and paste it below.\n\n"
                f"Login URL:\n{url}",
                title="myCampus login",
                border_style="cyan",
            )
        )
        if not no_browser:
            webbrowser.open(url)
        from prompt_toolkit.shortcuts import prompt

        token = prompt("Paste the moodlemobile:// address (or a token): ")
    try:
        parsed = parse_launch_response(token, settings.moodle_url, passport)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    if parsed.site_hash_valid is False:
        console.print(
            "[yellow]The site hash in the payload does not match this passport; continuing anyway.[/yellow]"
        )
    client = MoodleClient(settings.moodle_url, parsed.token)
    try:
        info = client.site_info()
    except (MoodleError, Exception) as exc:
        console.print(f"[red]The token was rejected by myCampus:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    path = save_token(settings, parsed, info)
    console.print(
        f"[green]Logged in as {info.get('fullname')} ({info.get('username')}).[/green] Token stored in {path}"
    )
    try:
        courses = client.user_courses(int(info["userid"]))
        console.print(f"{len(courses)} enrolled courses. Run `iu-agent moodle sync` to index them.")
    except MoodleError as exc:
        console.print(f"[yellow]Could not list courses: {exc}[/yellow]")


@moodle_app.command("status")
def moodle_status() -> None:
    """Show whether a myCampus token is stored and still valid."""
    settings = load_settings()
    print_moodle_status(settings)
    client = _moodle_client(settings)
    if client:
        try:
            info = client.site_info()
            console.print(
                f"[green]Token valid[/green] - site {info.get('sitename')}, Moodle {info.get('release')}"
            )
        except Exception as exc:
            console.print(f"[red]Token check failed:[/red] {exc}")


@moodle_app.command("courses")
def moodle_courses() -> None:
    """List the enrolled myCampus courses."""
    settings = load_settings()
    client = _moodle_client(settings)
    if client is None:
        console.print("Not logged in. Run `iu-agent moodle login`.")
        raise typer.Exit(code=1)
    info = client.site_info()
    table = Table(title=f"Courses of {info.get('fullname')}")
    table.add_column("id", justify="right")
    table.add_column("Course")
    table.add_column("Short name")
    for course in client.user_courses(int(info["userid"])):
        table.add_row(str(course["id"]), str(course.get("fullname")), str(course.get("shortname")))
    console.print(table)


@moodle_app.command("sync")
def moodle_sync(
    course: list[int] = typer.Option(None, "--course", help="Only sync these course ids (repeatable)."),
    force: bool = typer.Option(False, "--force", help="Re-index unchanged content too."),
) -> None:
    """Download course content from myCampus and index it."""
    settings = load_settings()
    client = _moodle_client(settings)
    if client is None:
        console.print("Not logged in. Run `iu-agent moodle login`.")
        raise typer.Exit(code=1)
    store = _open_store(settings)
    try:
        run_moodle_sync(settings, store, client, course_ids=list(course) if course else None, force=force)
    finally:
        store.close()


@moodle_app.command("logout")
def moodle_logout() -> None:
    """Delete the stored myCampus token."""
    settings = load_settings()
    console.print("Token removed." if clear_token(settings) else "No stored token.")


@moodle_app.command("check")
def moodle_check() -> None:
    """Show the public web-service configuration of the myCampus site (no login needed)."""
    settings = load_settings()
    config = MoodleClient.public_config(settings.moodle_url)
    table = Table(title=f"{config.get('sitename')} ({settings.moodle_url})", show_header=False)
    table.add_column("Key", style="bold")
    table.add_column("Value")
    login_types = {1: "username/password", 2: "browser", 3: "embedded browser (SSO)"}
    table.add_row("Web services enabled", "yes" if config.get("enablewebservices") else "no")
    table.add_row("Mobile service enabled", "yes" if config.get("enablemobilewebservice") else "no")
    table.add_row("Login type", login_types.get(config.get("typeoflogin"), str(config.get("typeoflogin"))))
    table.add_row(
        "Identity providers",
        ", ".join(p.get("name", "?") for p in config.get("identityproviders", [])) or "-",
    )
    table.add_row("Launch URL", str(config.get("launchurl")))
    console.print(table)


def _split_args(text: str) -> list[str]:
    try:
        return shlex.split(text)
    except ValueError:
        return text.split()


if __name__ == "__main__":  # pragma: no cover
    app()
