"""Tools available to the agent.

Read-only tools run immediately. ``write_file`` and ``run_shell`` pause the graph with a LangGraph
``interrupt`` so the CLI can ask the user for approval (like Claude Code's permission prompts).
"""

from __future__ import annotations

import difflib
import os
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, tool
from langgraph.types import interrupt

from iu_agent.config import Settings
from iu_agent.moodle.client import MoodleClient
from iu_agent.rag.ingest import Manifest
from iu_agent.rag.loaders import html_to_text, load_file
from iu_agent.rag.store import CourseVectorStore

MAX_TOOL_OUTPUT = 20_000


@dataclass
class AgentContext:
    settings: Settings
    manifest: Manifest
    store_factory: Callable[[], CourseVectorStore | None]
    workspace: Path
    moodle: MoodleClient | None = None
    auto_approve: set[str] = field(default_factory=set)
    approval_mode: str = "ask"  # ask | allow | deny  (non-interactive runs use allow/deny)


def _truncate(text: str, limit: int = MAX_TOOL_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more characters]"


def _format_hit(index: int, doc, score: float) -> str:
    meta = doc.metadata or {}
    where = []
    for key, label in (
        ("course", "course"),
        ("file_name", "file"),
        ("module", "module"),
        ("page", "p."),
        ("slide", "slide"),
    ):
        if meta.get(key) not in (None, ""):
            where.append(f"{label} {meta[key]}" if label in ("p.", "slide") else f"{label}: {meta[key]}")
    url = meta.get("url")
    location = " | ".join(where) + (f" | {url}" if url else "")
    body = doc.page_content
    if body.startswith("["):
        body = body.split("\n", 1)[1] if "\n" in body else body
    return f"### Result {index} (score {score:.3f}) - {location}\nsource: {meta.get('source')}\n{body}"


def resolve_course(name: str | None, manifest: Manifest) -> str | None:
    """Map a fuzzy course name / code typed by the model to an indexed course name."""
    if not name:
        return None
    courses = list(manifest.courses().keys())
    if name in courses:
        return name
    lowered = name.lower()
    for course in courses:
        if lowered in course.lower():
            return course
    codes = {}
    for entry in manifest.documents.values():
        if entry.get("course_code") and entry.get("course"):
            codes.setdefault(entry["course_code"].lower(), entry["course"])
    if lowered in codes:
        return codes[lowered]
    matches = difflib.get_close_matches(name, courses, n=1, cutoff=0.5)
    return matches[0] if matches else None


def _safe_path(ctx: AgentContext, raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = ctx.workspace / path
    path = path.resolve()
    if not ctx.settings.allow_outside_workspace:
        try:
            path.relative_to(ctx.workspace.resolve())
        except ValueError as exc:
            raise PermissionError(
                f"{path} is outside the workspace {ctx.workspace}. "
                "Set ALLOW_OUTSIDE_WORKSPACE=true to allow it."
            ) from exc
    return path


def _approve(ctx: AgentContext, kind: str, payload: dict[str, Any]) -> bool:
    if ctx.approval_mode == "allow" or kind in ctx.auto_approve:
        return True
    if ctx.approval_mode == "deny":
        return False
    decision = interrupt({"kind": kind, **payload})
    if isinstance(decision, dict):
        if decision.get("always"):
            ctx.auto_approve.add(kind)
        return bool(decision.get("approved"))
    return bool(decision)


def build_tools(ctx: AgentContext) -> list[BaseTool]:
    @tool
    def search_course_material(query: str, course: str | None = None, k: int | None = None) -> str:
        """Hybrid (semantic + keyword) search over the indexed IU course material: course books,
        lecture slides, exercises, exam preparation files and myCampus pages/announcements.
        Use it for every question about course content. `query` should be a specific phrase or
        question (German or English, matching the material). `course` optionally narrows the search
        to one course (name or course code such as DLMDSAM01). Returns the best matching chunks with
        their source, file name and page."""
        store = ctx.store_factory()
        if store is None:
            return "The vector store is not available (nothing indexed yet). Run `iu-agent ingest` first."
        resolved = resolve_course(course, ctx.manifest)
        limit = k or ctx.settings.retrieval_k
        try:
            hits = store.search(query, k=limit, course=resolved)
        except Exception as exc:
            return f"Search failed: {type(exc).__name__}: {exc}"
        if not hits:
            scope = f" in course '{resolved or course}'" if course else ""
            return f"No matching material found{scope}. Try other keywords or drop the course filter."
        header = f"Course filter: {resolved}\n" if resolved else ""
        if course and not resolved:
            header = f"Course '{course}' is not indexed, searched everything instead.\n"
        return _truncate(
            header + "\n\n".join(_format_hit(i, doc, score) for i, (doc, score) in enumerate(hits, 1))
        )

    @tool
    def list_courses() -> str:
        """List the indexed courses with the number of indexed documents per course."""
        courses = ctx.manifest.courses()
        if not courses:
            return (
                "Nothing is indexed yet. Run `iu-agent ingest` (OneDrive folder) or `iu-agent moodle sync`."
            )
        lines = [f"- {name}: {count} documents" for name, count in courses.items()]
        return f"{len(courses)} indexed courses ({len(ctx.manifest.documents)} documents):\n" + "\n".join(
            lines
        )

    @tool
    def list_documents(course: str | None = None, contains: str | None = None) -> str:
        """List indexed documents (file names and sources), optionally filtered by course and/or a
        substring of the file name. Use it to find the exact `source` for read_document."""
        resolved = resolve_course(course, ctx.manifest) if course else None
        rows = []
        for source, entry in ctx.manifest.documents.items():
            if resolved and entry.get("course") != resolved:
                continue
            if contains and contains.lower() not in (entry.get("file_name", "") + source).lower():
                continue
            rows.append(
                f"- {entry.get('file_name')}  [{entry.get('course')}]  "
                f"source={source}  chunks={entry.get('chunks')}"
            )
        if not rows:
            return "No documents match."
        return _truncate(f"{len(rows)} documents:\n" + "\n".join(rows[:300]))

    @tool
    def read_document(source: str, start_page: int = 1, end_page: int | None = None) -> str:
        """Read the full text of an indexed document (or a page range for PDFs / PPTX). `source` is
        the source id from search results or list_documents, or a unique part of the file name.
        Returns at most 20k characters; use page ranges for long books."""
        matches = ctx.manifest.find(source)
        exact = [m for m in matches if m[0] == source]
        if exact:
            matches = exact
        if not matches:
            return f"No indexed document matches '{source}'. Use list_documents to find it."
        if len(matches) > 1:
            options = "\n".join(f"- {s}" for s, _ in matches[:20])
            return f"'{source}' is ambiguous, choose one of:\n{options}"
        key, entry = matches[0]
        path = entry.get("path")
        if not path:
            rel = key.split(":", 1)[1] if key.startswith("onedrive:") else None
            path = str(ctx.settings.iu_docs_path / rel) if rel else None
        if not path or not Path(path).exists():
            return f"The file for {key} is not available locally (Moodle HTML content is only searchable)."
        try:
            sections = load_file(Path(path))
        except Exception as exc:
            return f"Could not read {path}: {exc}"
        selected = []
        for section in sections:
            locator = section.meta.get("page") or section.meta.get("slide")
            if locator is not None:
                if locator < start_page or (end_page is not None and locator > end_page):
                    continue
            selected.append(section)
        if not selected:
            return "No content in that page range."
        parts = []
        for section in selected:
            locator = section.meta.get("page") or section.meta.get("slide")
            label = f"--- page {locator} ---\n" if locator is not None else ""
            parts.append(label + section.text)
        return _truncate(f"{entry.get('file_name')} ({entry.get('course')})\n\n" + "\n\n".join(parts))

    @tool
    def list_directory(path: str = ".") -> str:
        """List files and folders below a path in the workspace (non-recursive)."""
        try:
            target = _safe_path(ctx, path)
        except PermissionError as exc:
            return str(exc)
        if not target.exists():
            return f"{target} does not exist."
        if target.is_file():
            return f"{target} is a file ({target.stat().st_size} bytes)."
        entries = []
        for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if child.name in {".git", "__pycache__", ".venv", "node_modules"}:
                continue
            suffix = "/" if child.is_dir() else f"  ({child.stat().st_size} bytes)"
            entries.append(f"{child.name}{suffix}")
        return _truncate(f"{target}\n" + "\n".join(entries) if entries else f"{target} is empty.")

    @tool
    def read_file(path: str, start_line: int = 1, end_line: int | None = None) -> str:
        """Read a text file from the workspace, optionally only a line range (1-based, inclusive)."""
        try:
            target = _safe_path(ctx, path)
        except PermissionError as exc:
            return str(exc)
        if not target.is_file():
            return f"{target} is not a file."
        try:
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return f"Cannot read {target}: {exc}"
        end = end_line or len(lines)
        chosen = lines[max(start_line - 1, 0) : end]
        numbered = "\n".join(f"{start_line + i:5d}  {line}" for i, line in enumerate(chosen))
        return _truncate(f"{target} ({len(lines)} lines)\n{numbered}")

    @tool
    def write_file(path: str, content: str) -> str:
        """Create or overwrite a text file in the workspace with the given content. The user is
        asked for approval before anything is written."""
        try:
            target = _safe_path(ctx, path)
        except PermissionError as exc:
            return str(exc)
        exists = target.exists()
        old = target.read_text(encoding="utf-8", errors="replace") if exists and target.is_file() else ""
        diff = "\n".join(
            difflib.unified_diff(
                old.splitlines(), content.splitlines(), fromfile=f"a/{path}", tofile=f"b/{path}", lineterm=""
            )
        )
        if not _approve(
            ctx, "write_file", {"path": str(target), "exists": exists, "diff": diff, "content": content}
        ):
            return "The user declined the file write. Do not retry without asking what they prefer."
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return f"Wrote {len(content)} characters to {target} ({'overwritten' if exists else 'created'})."

    @tool
    def run_shell(command: str, timeout_seconds: int = 120) -> str:
        """Run a shell command in the workspace (PowerShell on Windows, bash elsewhere) and return
        stdout, stderr and the exit code. The user is asked for approval before it runs."""
        if not _approve(ctx, "run_shell", {"command": command, "cwd": str(ctx.workspace)}):
            return "The user declined to run the command."
        if sys.platform == "win32":
            argv = ["powershell", "-NoProfile", "-NonInteractive", "-Command", command]
        else:
            argv = ["bash", "-lc", command]
        try:
            completed = subprocess.run(
                argv,
                cwd=str(ctx.workspace),
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
        except subprocess.TimeoutExpired:
            return f"Command timed out after {timeout_seconds}s."
        except OSError as exc:
            return f"Could not start the command: {exc}"
        output = f"exit code: {completed.returncode}\n"
        if completed.stdout:
            output += f"stdout:\n{completed.stdout}\n"
        if completed.stderr:
            output += f"stderr:\n{completed.stderr}\n"
        return _truncate(output)

    @tool
    def fetch_url(url: str, max_chars: int = 8000) -> str:
        """Fetch a public web page and return its text content (HTML is converted to text)."""
        import httpx

        try:
            response = httpx.get(
                url, follow_redirects=True, timeout=30.0, headers={"User-Agent": "iu-campus-agent/0.1"}
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            return f"Fetch failed: {exc}"
        content_type = response.headers.get("content-type", "")
        text = html_to_text(response.text) if "html" in content_type else response.text
        return _truncate(text, max_chars)

    tools: list[BaseTool] = [
        search_course_material,
        list_courses,
        list_documents,
        read_document,
        list_directory,
        read_file,
        write_file,
        run_shell,
        fetch_url,
    ]

    if ctx.moodle is not None:
        moodle = ctx.moodle

        @tool
        def moodle_list_courses() -> str:
            """List the courses the student is enrolled in on myCampus (live from the Moodle API)."""
            try:
                info = moodle.site_info()
                courses = moodle.user_courses(int(info["userid"]))
            except Exception as exc:
                return f"myCampus request failed: {exc}"
            lines = [f"- id {c['id']}: {c.get('fullname')} ({c.get('shortname')})" for c in courses]
            return "\n".join(lines) if lines else "No enrolled courses returned."

        @tool
        def moodle_course_contents(course_id: int) -> str:
            """Show the section / activity structure of one myCampus course (live from the Moodle
            API), including activity types and links. Use moodle_list_courses to find the id."""
            try:
                sections = moodle.course_contents(course_id)
            except Exception as exc:
                return f"myCampus request failed: {exc}"
            lines = []
            for section in sections:
                lines.append(f"## {section.get('name')}")
                for module in section.get("modules", []) or []:
                    files = [
                        c.get("filename") for c in module.get("contents", []) or [] if c.get("type") == "file"
                    ]
                    extra = f" files: {', '.join(files)}" if files else ""
                    lines.append(
                        f"- [{module.get('modname')}] {module.get('name')} {module.get('url') or ''}{extra}"
                    )
            return _truncate("\n".join(lines))

        tools.extend([moodle_list_courses, moodle_course_contents])

    return tools
