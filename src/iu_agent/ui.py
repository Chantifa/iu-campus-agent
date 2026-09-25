"""Terminal UI helpers: banner, streaming Markdown renderer, tool-call display and approval prompts."""

from __future__ import annotations

import io
import json
import sys
import time
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from iu_agent.models import PROVIDER_LABELS, ModelInfo, ModelRef

LIVE_WINDOW = 8  # number of trailing lines that stay "live" while streaming


def make_console() -> Console:
    return Console(highlight=False, soft_wrap=False)


def glyph(console: Console, unicode_char: str, ascii_fallback: str) -> str:
    encoding = (console.encoding or "").lower()
    return unicode_char if "utf" in encoding else ascii_fallback


def _short(value: Any, limit: int = 90) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str)
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 1] + "…"


class StreamRenderer:
    """Render streamed Markdown progressively.

    Finished lines are printed permanently above a small live region that holds the last few
    lines; this keeps long answers scrollable and avoids flicker or duplicated output.
    """

    def __init__(self, console: Console, min_interval: float = 0.08) -> None:
        self.console = console
        self.min_interval = min_interval
        self.buffer = ""
        self.printed = 0
        self.live: Live | None = None
        self.status = None
        self.last_update = 0.0

    def thinking(self, label: str = "thinking") -> None:
        if self.status is None and self.live is None:
            self.status = self.console.status(f"[dim]{label}…[/dim]", spinner="dots")
            self.status.start()

    def _stop_status(self) -> None:
        if self.status is not None:
            self.status.stop()
            self.status = None

    def _render_lines(self, text: str) -> list[str]:
        buffer = io.StringIO()
        scratch = Console(
            file=buffer,
            force_terminal=True,
            color_system="truecolor",
            width=self.console.width,
            legacy_windows=False,
            highlight=False,
        )
        scratch.print(Markdown(text))
        return buffer.getvalue().rstrip("\n").split("\n")

    def append(self, text: str) -> None:
        if not text:
            return
        self._stop_status()
        self.buffer += text
        now = time.monotonic()
        if now - self.last_update < self.min_interval:
            return
        self.last_update = now
        self._update(final=False)

    def _update(self, final: bool) -> None:
        lines = self._render_lines(self.buffer)
        if self.live is None:
            self.live = Live(Text(""), console=self.console, refresh_per_second=12, transient=True)
            self.live.start()
        stable = len(lines) if final else max(0, len(lines) - LIVE_WINDOW)
        if stable > self.printed:
            self.live.console.print(Text.from_ansi("\n".join(lines[self.printed : stable])))
            self.printed = stable
        tail = lines[self.printed :]
        self.live.update(Text.from_ansi("\n".join(tail)) if tail and not final else Text(""))

    def finish(self) -> str:
        self._stop_status()
        if self.buffer:
            self._update(final=True)
        if self.live is not None:
            self.live.stop()
            self.live = None
        text = self.buffer
        self.buffer = ""
        self.printed = 0
        if text:
            self.console.print()
        return text


def render_tool_call(console: Console, name: str, args: dict[str, Any]) -> None:
    preview = ", ".join(f"{key}={_short(value)}" for key, value in (args or {}).items())
    console.print(
        Text.assemble(
            (f"{glyph(console, '⚙', '*')} ", "bold cyan"), (name, "bold cyan"), (f"({preview})", "cyan")
        )
    )


def render_tool_result(console: Console, name: str, content: Any) -> None:
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
    lines = [line for line in text.strip().splitlines() if line.strip()]
    first = lines[0] if lines else "(empty)"
    more = f" (+{len(lines) - 1} lines)" if len(lines) > 1 else ""
    console.print(Text(f"  {glyph(console, '↳', '->')} {_short(first, 110)}{more}", style="dim"))


def render_approval_request(console: Console, request: dict[str, Any]) -> None:
    kind = request.get("kind")
    if kind == "run_shell":
        console.print(
            Panel(
                Syntax(str(request.get("command", "")), "powershell", word_wrap=True),
                title="[yellow]Run this command?[/yellow]",
                subtitle=f"cwd: {request.get('cwd', '')}",
                border_style="yellow",
            )
        )
    elif kind == "write_file":
        body = request.get("diff") or request.get("content") or ""
        language = "diff" if request.get("diff") else "text"
        if len(body) > 6000:
            body = body[:6000] + "\n... (truncated)"
        title = "Overwrite file?" if request.get("exists") else "Create file?"
        console.print(
            Panel(
                Syntax(body, language, word_wrap=True),
                title=f"[yellow]{title}[/yellow] {request.get('path', '')}",
                border_style="yellow",
            )
        )
    else:
        console.print(
            Panel(json.dumps(request, indent=1, default=str), title="Approve?", border_style="yellow")
        )


def interactive_terminal() -> bool:
    """True when a human can answer prompts (stdin and stdout are terminals)."""
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except (AttributeError, ValueError):
        return False


def ask_approval(console: Console, request: dict[str, Any]) -> dict[str, bool]:
    from prompt_toolkit.shortcuts import prompt

    render_approval_request(console, request)
    if not interactive_terminal():
        console.print(Text("  declined (no interactive terminal)", style="red"))
        return {"approved": False, "always": False}
    try:
        answer = prompt("  Allow? [y]es / [n]o / [a]lways for this session: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        answer = "n"
    approved = answer in ("y", "yes", "a", "always", "ja", "j")
    always = answer in ("a", "always")
    console.print(Text("  approved" if approved else "  declined", style="green" if approved else "red"))
    return {"approved": approved, "always": always}


def models_table(models: list[ModelInfo], current: ModelRef | None = None) -> Table:
    table = Table(title="Available models", show_lines=False, header_style="bold")
    table.add_column("#", justify="right", style="dim")
    table.add_column("Model id")
    table.add_column("Provider")
    table.add_column("Note", style="dim")
    for index, info in enumerate(models, start=1):
        marker = " (current)" if current and info.ref == current else ""
        style = "bold green" if marker else ""
        table.add_row(
            str(index),
            Text(info.ref.id + marker, style=style),
            PROVIDER_LABELS.get(info.ref.provider, info.ref.provider),
            info.note,
        )
    return table


def banner(
    console: Console,
    *,
    version: str,
    model: ModelRef | None,
    store_location: str,
    documents: int,
    chunks: int,
    courses: int,
    moodle: str,
    workspace: str,
) -> None:
    model_text = model.id if model else "none configured (search only)"
    lines = [
        f"[bold]Model[/bold]      {model_text}",
        f"[bold]Vector DB[/bold]  {store_location}",
        f"[bold]Index[/bold]      {documents} documents, {chunks} chunks, {courses} courses",
        f"[bold]myCampus[/bold]   {moodle}",
        f"[bold]Workspace[/bold]  {workspace}",
        "",
        "[dim]Type a question or task. /help lists commands, /model switches the LLM, Ctrl+D exits.[/dim]",
    ]
    console.print(
        Panel(
            "\n".join(lines), title=f"[bold cyan]IU Campus Agent[/bold cyan] v{version}", border_style="cyan"
        )
    )


HELP_TEXT = """\
[bold]Commands[/bold]
  /help                 show this help
  /model [id]           choose a model interactively or switch to <provider:model>
  /models [live]        list models (add 'live' to query the provider APIs)
  /search <query>       raw retrieval from the vector store (no LLM)
  /courses              indexed courses
  /ingest [path]        (re)index the OneDrive/IU folder or another folder
  /moodle sync|status   sync myCampus content / show connection status
  /clear                start a new conversation thread
  /status               configuration and index statistics
  /exit                 quit (Ctrl+D works too)

[bold]Editing[/bold]
  Enter sends, Alt+Enter inserts a newline, Ctrl+C clears the input, Up/Down browse history.
"""
