"""Pull course content from myCampus (Moodle web services) and feed it into the ingestion pipeline.

What is collected per enrolled course:

* files attached to resources / folders / assignments (PDF, DOCX, PPTX ... via the loaders),
* pages, book chapters, labels and module descriptions (HTML -> text),
* assignment descriptions with due dates,
* announcement forum posts (``MOODLE_SYNC_FORUMS=news``; ``all`` for every forum, ``none`` to skip),
* URL activities as short link records (optionally fetched, ``MOODLE_FETCH_URLS=true``).

Videos, SCORM packages, quizzes and external LTI tools cannot be exported through the API and are
listed as skipped.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from iu_agent.config import Settings
from iu_agent.moodle.client import MoodleClient, MoodleError
from iu_agent.rag.ingest import ORIGIN_MOODLE, Ingestor, IngestReport, ProgressCallback, SourceDocument
from iu_agent.rag.loaders import SUPPORTED_EXTENSIONS, Section, html_to_text

_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")
_MIN_HTML_CHARS = 40


def safe_name(name: str, limit: int = 120) -> str:
    cleaned = _SAFE_RE.sub("_", name).strip("._")
    return (cleaned or "file")[:limit]


@dataclass
class SyncPlan:
    documents: list[SourceDocument] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class MoodleSync:
    def __init__(
        self,
        settings: Settings,
        client: MoodleClient,
        ingestor: Ingestor,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.settings = settings
        self.client = client
        self.ingestor = ingestor
        self.log = log or (lambda _msg: None)
        self.cache_dir = settings.moodle_cache_dir

    # ------------------------------------------------------------------ discovery
    def site_info(self) -> dict[str, Any]:
        return self.client.site_info()

    def courses(self) -> list[dict[str, Any]]:
        info = self.site_info()
        return self.client.user_courses(int(info["userid"]))

    # ------------------------------------------------------------------ collection
    def collect(self, course_ids: list[int] | None = None) -> SyncPlan:
        plan = SyncPlan()
        for course in self.courses():
            if course_ids and int(course["id"]) not in course_ids:
                continue
            self.log(f"Collecting {course.get('fullname')} (id {course['id']})")
            try:
                self._collect_course(course, plan)
            except MoodleError as exc:
                plan.warnings.append(f"{course.get('fullname')}: {exc}")
        return plan

    def _course_meta(self, course: dict[str, Any]) -> dict[str, Any]:
        return {
            "origin": ORIGIN_MOODLE,
            "course": course.get("fullname") or course.get("shortname") or str(course["id"]),
            "course_short": course.get("shortname"),
            "course_id": int(course["id"]),
            "course_url": f"{self.client.base_url}/course/view.php?id={course['id']}",
        }

    def _collect_course(self, course: dict[str, Any], plan: SyncPlan) -> None:
        course_id = int(course["id"])
        base = self._course_meta(course)
        sections = self.client.course_contents(course_id)
        for section in sections:
            section_name = section.get("name") or f"Section {section.get('section')}"
            summary = section.get("summary") or ""
            if len(summary) > _MIN_HTML_CHARS:
                plan.documents.append(
                    self._html_document(
                        source=f"moodle:{course_id}/section-{section.get('id')}",
                        html=summary,
                        meta={
                            **base,
                            "section": section_name,
                            "module": section_name,
                            "module_type": "section",
                        },
                        fingerprint=_hash(summary),
                        file_name=f"{section_name}.html",
                    )
                )
            for module in section.get("modules", []) or []:
                self._collect_module(module, section_name, base, plan)
        self._collect_pages(course_id, base, plan)
        self._collect_assignments(course_id, base, plan)
        self._collect_forums(course_id, base, plan)

    def _collect_module(
        self, module: dict[str, Any], section_name: str, base: dict[str, Any], plan: SyncPlan
    ) -> None:
        modname = module.get("modname", "")
        cmid = module.get("id")
        meta = {
            **base,
            "section": section_name,
            "module": module.get("name"),
            "module_type": modname,
            "url": module.get("url"),
            "cmid": cmid,
        }
        description = module.get("description") or ""
        if len(description) > _MIN_HTML_CHARS and modname not in ("page", "assign"):
            plan.documents.append(
                self._html_document(
                    source=f"moodle:{base['course_id']}/{cmid}/description",
                    html=description,
                    meta=meta,
                    fingerprint=_hash(description),
                    file_name=f"{safe_name(module.get('name', 'module'))}.html",
                )
            )
        if modname in ("page", "forum", "assign"):
            return  # handled by dedicated web-service functions
        if modname in ("quiz", "scorm", "lti", "kalvidres", "hvp", "h5pactivity", "feedback", "choice"):
            plan.skipped.append(f"{base['course']}: {module.get('name')} ({modname})")
            return
        book_titles = _book_chapter_titles(module) if modname == "book" else {}
        for content in module.get("contents", []) or []:
            ctype = content.get("type")
            fileurl = content.get("fileurl")
            filename = content.get("filename") or "file"
            if ctype == "file" and fileurl:
                extension = Path(filename).suffix.lower()
                if extension not in SUPPORTED_EXTENSIONS:
                    plan.skipped.append(
                        f"{base['course']}: {module.get('name')} / {filename} (unsupported type)"
                    )
                    continue
                target = self.cache_dir / str(base["course_id"]) / f"{cmid}_{safe_name(filename)}"
                plan.documents.append(
                    SourceDocument(
                        source=f"moodle:{base['course_id']}/{cmid}/{filename}",
                        fingerprint=f"{content.get('timemodified')}:{content.get('filesize')}",
                        meta={
                            **meta,
                            "file_name": filename,
                            "file_type": extension.lstrip("."),
                            "fileurl": fileurl,
                        },
                        fetch=self._downloader(fileurl, target),
                    )
                )
            elif ctype == "content" and fileurl:
                if filename == "structure":
                    continue
                chapter_id = (content.get("filepath") or "/").strip("/")
                title = book_titles.get(chapter_id) or content.get("content") or module.get("name")
                chapter_meta = {**meta, "chapter": title, "file_name": f"{safe_name(str(title))}.html"}
                plan.documents.append(
                    SourceDocument(
                        source=f"moodle:{base['course_id']}/{cmid}/chapter-{chapter_id or filename}",
                        fingerprint=f"{content.get('timemodified')}:{content.get('filesize')}",
                        meta=chapter_meta,
                        fetch=self._html_fetcher(fileurl, title=str(title)),
                    )
                )
            elif ctype == "url" and fileurl:
                link_meta = {
                    **meta,
                    "file_name": f"{safe_name(module.get('name', 'link'))}.url",
                    "external_url": fileurl,
                }
                details = html_to_text(description) if description else ""
                text = f"# {module.get('name')}\n\nLink: {fileurl}\n\n{details}"
                sections = [Section(text.strip())]
                if self.settings.moodle_fetch_urls:
                    plan.documents.append(
                        SourceDocument(
                            source=f"moodle:{base['course_id']}/{cmid}/url",
                            fingerprint=_hash(fileurl + description),
                            meta=link_meta,
                            fetch=self._external_fetcher(fileurl, fallback=sections),
                        )
                    )
                else:
                    plan.documents.append(
                        SourceDocument(
                            source=f"moodle:{base['course_id']}/{cmid}/url",
                            fingerprint=_hash(fileurl + description),
                            meta=link_meta,
                            sections=sections,
                        )
                    )

    def _collect_pages(self, course_id: int, base: dict[str, Any], plan: SyncPlan) -> None:
        try:
            pages = self.client.pages([course_id])
        except MoodleError as exc:
            plan.warnings.append(f"{base['course']}: pages unavailable ({exc})")
            return
        for page in pages:
            html = f"<h1>{page.get('name', '')}</h1>{page.get('intro') or ''}{page.get('content') or ''}"
            plan.documents.append(
                self._html_document(
                    source=f"moodle:{course_id}/{page.get('coursemodule')}/page",
                    html=html,
                    meta={
                        **base,
                        "module": page.get("name"),
                        "module_type": "page",
                        "url": f"{self.client.base_url}/mod/page/view.php?id={page.get('coursemodule')}",
                        "cmid": page.get("coursemodule"),
                    },
                    fingerprint=f"{page.get('timemodified')}:{_hash(html)}",
                    file_name=f"{safe_name(page.get('name', 'page'))}.html",
                )
            )

    def _collect_assignments(self, course_id: int, base: dict[str, Any], plan: SyncPlan) -> None:
        try:
            assignments = self.client.assignments([course_id])
        except MoodleError as exc:
            plan.warnings.append(f"{base['course']}: assignments unavailable ({exc})")
            return
        for assignment in assignments:
            due = assignment.get("duedate")
            due_text = time.strftime("%Y-%m-%d %H:%M", time.localtime(due)) if due else "none"
            html = (
                f"<h1>Assignment: {assignment.get('name', '')}</h1>"
                f"<p>Due date: {due_text}</p>{assignment.get('intro') or ''}"
            )
            plan.documents.append(
                self._html_document(
                    source=f"moodle:{course_id}/{assignment.get('cmid')}/assignment",
                    html=html,
                    meta={
                        **base,
                        "module": assignment.get("name"),
                        "module_type": "assign",
                        "url": f"{self.client.base_url}/mod/assign/view.php?id={assignment.get('cmid')}",
                        "cmid": assignment.get("cmid"),
                        "due_date": due_text,
                    },
                    fingerprint=f"{assignment.get('timemodified')}:{_hash(html)}",
                    file_name=f"{safe_name(assignment.get('name', 'assignment'))}.html",
                )
            )

    def _collect_forums(self, course_id: int, base: dict[str, Any], plan: SyncPlan) -> None:
        mode = (self.settings.moodle_sync_forums or "news").lower()
        if mode == "none":
            return
        try:
            forums = self.client.forums([course_id])
        except MoodleError as exc:
            plan.warnings.append(f"{base['course']}: forums unavailable ({exc})")
            return
        for forum in forums:
            if mode == "news" and forum.get("type") != "news":
                continue
            try:
                discussions = self.client.discussions(int(forum["id"]))
            except MoodleError as exc:
                plan.warnings.append(f"{base['course']}: forum {forum.get('name')} unavailable ({exc})")
                continue
            for discussion in discussions:
                created = discussion.get("created") or discussion.get("timemodified")
                stamp = time.strftime("%Y-%m-%d", time.localtime(created)) if created else ""
                html = (
                    f"<h1>{discussion.get('name', '')}</h1>"
                    f"<p>{forum.get('name')} - {discussion.get('userfullname', '')} {stamp}</p>"
                    f"{discussion.get('message') or ''}"
                )
                plan.documents.append(
                    self._html_document(
                        source=f"moodle:{course_id}/forum-{forum['id']}/discussion-{discussion.get('discussion')}",
                        html=html,
                        meta={
                            **base,
                            "module": forum.get("name"),
                            "module_type": "forum",
                            "url": (
                                f"{self.client.base_url}/mod/forum/discuss.php"
                                f"?d={discussion.get('discussion')}"
                            ),
                            "posted": stamp,
                        },
                        fingerprint=f"{discussion.get('timemodified')}:{_hash(html)}",
                        file_name=f"{safe_name(discussion.get('name', 'post'))}.html",
                    )
                )

    # ------------------------------------------------------------------ helpers
    def _html_document(
        self, *, source: str, html: str, meta: dict[str, Any], fingerprint: str, file_name: str
    ) -> SourceDocument:
        text = html_to_text(html)
        return SourceDocument(
            source=source,
            fingerprint=fingerprint,
            meta={**meta, "file_name": file_name, "file_type": "html"},
            sections=[Section(text)] if text else [],
        )

    def _downloader(self, fileurl: str, target: Path) -> Callable[[], Path]:
        def fetch() -> Path:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(self.client.download(fileurl))
            return target

        return fetch

    def _html_fetcher(self, fileurl: str, title: str) -> Callable[[], list[Section]]:
        def fetch() -> list[Section]:
            html = self.client.download(fileurl).decode("utf-8", errors="replace")
            text = html_to_text(f"<h1>{title}</h1>{html}")
            return [Section(text)] if text else []

        return fetch

    def _external_fetcher(self, url: str, fallback: list[Section]) -> Callable[[], list[Section]]:
        def fetch() -> list[Section]:
            try:
                response = httpx.get(url, follow_redirects=True, timeout=30.0)
                response.raise_for_status()
                if "html" in response.headers.get("content-type", ""):
                    text = html_to_text(response.text)
                    if text:
                        return [Section(f"Link: {url}\n\n{text[:20000]}")]
            except httpx.HTTPError:
                pass
            return fallback

        return fetch

    # ------------------------------------------------------------------ run
    def run(
        self,
        course_ids: list[int] | None = None,
        *,
        force: bool = False,
        prune: bool = True,
        on_progress: ProgressCallback | None = None,
    ) -> tuple[IngestReport, SyncPlan]:
        plan = self.collect(course_ids)
        report = self.ingestor.ingest(plan.documents, force=force, on_progress=on_progress)
        if prune and not course_ids:
            report.removed = self.ingestor.prune(ORIGIN_MOODLE, {d.source for d in plan.documents})
        return report, plan


def _book_chapter_titles(module: dict[str, Any]) -> dict[str, str]:
    for content in module.get("contents", []) or []:
        if content.get("filename") == "structure" and content.get("content"):
            try:
                structure = json.loads(content["content"])
            except (TypeError, ValueError):
                return {}
            titles: dict[str, str] = {}
            for chapter in structure:
                href = str(chapter.get("href", "")).split("/")[0]
                titles[href] = chapter.get("title", "")
                for sub in chapter.get("subitems", []) or []:
                    titles[str(sub.get("href", "")).split("/")[0]] = sub.get("title", "")
            return titles
    return {}


def _hash(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8", errors="replace")).hexdigest()  # noqa: S324
