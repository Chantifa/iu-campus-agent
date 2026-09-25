"""Incremental ingestion of course material into the vector store.

A JSON manifest (``DATA_DIR/manifest.json``) remembers a fingerprint per source document so that
re-running the ingestion only re-embeds new or changed files and removes chunks of deleted ones.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from iu_agent.config import Settings
from iu_agent.rag.chunking import chunk_sections
from iu_agent.rag.loaders import Section, is_supported, load_file
from iu_agent.rag.store import CourseVectorStore

ORIGIN_ONEDRIVE = "onedrive"
ORIGIN_MOODLE = "moodle"

COURSE_CODE_RE = re.compile(r"\b([A-Z]{3,}[A-Z0-9]*\d{2,3})\b")
_SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "__pycache__", ".idea", ".ipynb_checkpoints"}

ProgressCallback = Callable[[int, int, str], None]


@dataclass
class SourceDocument:
    """One logical document to index. Content comes from ``path``, ``sections`` or a lazy ``fetch``."""

    source: str
    fingerprint: str
    meta: dict[str, Any]
    path: Path | None = None
    sections: list[Section] | None = None
    fetch: Callable[[], Path | list[Section]] | None = None

    def load(self) -> tuple[list[Section], Path | None]:
        if self.sections is not None:
            return self.sections, self.path
        target = self.path
        if target is None and self.fetch is not None:
            result = self.fetch()
            if isinstance(result, list):
                return result, None
            target = Path(result)
        if target is None:
            raise ValueError(f"{self.source}: nothing to load")
        return load_file(target), target


@dataclass
class IngestReport:
    added: int = 0
    updated: int = 0
    skipped: int = 0
    removed: int = 0
    chunks: int = 0
    failed: list[tuple[str, str]] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"added {self.added}, updated {self.updated}, unchanged {self.skipped}, removed {self.removed}, "
            f"chunks written {self.chunks}, failed {len(self.failed)}"
        )


class Manifest:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data: dict[str, Any] = {"version": 1, "documents": {}}
        if path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                pass
        self.data.setdefault("documents", {})

    @property
    def documents(self) -> dict[str, dict[str, Any]]:
        return self.data["documents"]

    def get(self, source: str) -> dict[str, Any] | None:
        return self.documents.get(source)

    def set(self, source: str, entry: dict[str, Any]) -> None:
        self.documents[source] = entry

    def remove(self, source: str) -> None:
        self.documents.pop(source, None)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.path)

    def sources(self, origin: str | None = None) -> list[str]:
        return [s for s, e in self.documents.items() if origin is None or e.get("origin") == origin]

    def courses(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for entry in self.documents.values():
            course = entry.get("course") or "General"
            counts[course] = counts.get(course, 0) + 1
        return dict(sorted(counts.items()))

    def total_chunks(self) -> int:
        return sum(int(e.get("chunks", 0)) for e in self.documents.values())

    def find(self, needle: str) -> list[tuple[str, dict[str, Any]]]:
        needle = needle.lower()
        return [
            (source, entry)
            for source, entry in self.documents.items()
            if needle in source.lower() or needle in str(entry.get("file_name", "")).lower()
        ]


# ----------------------------------------------------------------------------- folder scanning
def file_fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _matches(rel_posix: str, patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        pattern = pattern.replace("\\", "/").strip("/")
        if not pattern:
            continue
        normalised = pattern.replace("**", "*")
        if fnmatch.fnmatch(rel_posix, normalised) or fnmatch.fnmatch(rel_posix, normalised + "/*"):
            return True
    return False


def iter_source_files(
    root: Path,
    include: Iterable[str] = (),
    exclude: Iterable[str] = (),
    max_bytes: int | None = None,
) -> list[Path]:
    include, exclude = list(include), list(exclude)
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d not in _SKIP_DIRS)
        for name in sorted(filenames):
            path = Path(dirpath) / name
            if name.startswith((".", "~$")) or not is_supported(path):
                continue
            rel = path.relative_to(root).as_posix()
            if include and not _matches(rel, include):
                continue
            if exclude and _matches(rel, exclude):
                continue
            if max_bytes is not None:
                try:
                    if path.stat().st_size > max_bytes:
                        continue
                except OSError:
                    continue
            files.append(path)
    return files


def infer_file_meta(root: Path, path: Path) -> dict[str, Any]:
    rel = path.relative_to(root)
    parts = rel.parts
    semester: str | None = None
    course = "General"
    if len(parts) > 1:
        if parts[0].lower().startswith("semester") and len(parts) > 2:
            semester, course = parts[0], parts[1]
        else:
            course = parts[0]
    code_match = COURSE_CODE_RE.search(path.stem) or COURSE_CODE_RE.search(str(rel))
    meta: dict[str, Any] = {
        "origin": ORIGIN_ONEDRIVE,
        "file_name": path.name,
        "rel_path": rel.as_posix(),
        "course": course,
        "file_type": path.suffix.lower().lstrip("."),
    }
    if semester:
        meta["semester"] = semester
    if code_match:
        meta["course_code"] = code_match.group(1)
    return meta


# ----------------------------------------------------------------------------- ingestion
class Ingestor:
    def __init__(
        self, settings: Settings, store: CourseVectorStore, manifest: Manifest | None = None
    ) -> None:
        self.settings = settings
        self.store = store
        self.manifest = manifest or Manifest(settings.manifest_path)

    def ingest(
        self,
        documents: list[SourceDocument],
        *,
        force: bool = False,
        on_progress: ProgressCallback | None = None,
    ) -> IngestReport:
        report = IngestReport()
        total = len(documents)
        for index, document in enumerate(documents, start=1):
            if on_progress:
                on_progress(index, total, document.meta.get("file_name") or document.source)
            existing = self.manifest.get(document.source)
            if existing and existing.get("fingerprint") == document.fingerprint and not force:
                report.skipped += 1
                continue
            try:
                sections, loaded_path = document.load()
                chunks = chunk_sections(
                    sections,
                    source=document.source,
                    base_meta=document.meta,
                    chunk_size=self.settings.chunk_size,
                    chunk_overlap=self.settings.chunk_overlap,
                )
                if existing:
                    self.store.delete_source(document.source)
                self.store.add_documents(chunks, ids=[c.metadata["id"] for c in chunks])
            except Exception as exc:  # keep going, report at the end
                report.failed.append((document.source, f"{type(exc).__name__}: {exc}"))
                continue
            entry = {
                "fingerprint": document.fingerprint,
                "chunks": len(chunks),
                "origin": document.meta.get("origin"),
                "course": document.meta.get("course"),
                "file_name": document.meta.get("file_name"),
                "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            if loaded_path is not None:
                entry["path"] = str(loaded_path)
            for key in ("course_id", "module", "url", "semester", "course_code"):
                if document.meta.get(key) is not None:
                    entry[key] = document.meta[key]
            self.manifest.set(document.source, entry)
            report.chunks += len(chunks)
            if existing:
                report.updated += 1
            else:
                report.added += 1
            if index % 10 == 0:
                self.manifest.save()
        self.manifest.save()
        return report

    def prune(self, origin: str, keep: set[str]) -> int:
        removed = 0
        for source in self.manifest.sources(origin):
            if source not in keep:
                self.store.delete_source(source)
                self.manifest.remove(source)
                removed += 1
        if removed:
            self.manifest.save()
        return removed

    def remove_all(self) -> None:
        self.store.reset()
        self.manifest.data["documents"] = {}
        self.manifest.save()

    # ------------------------------------------------------------------ OneDrive / local folder
    def collect_folder(self, root: Path | None = None) -> list[SourceDocument]:
        root = Path(root or self.settings.iu_docs_path).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Documents folder not found: {root}")
        # Course names and source keys stay relative to the IU folder, so indexing a single
        # sub folder (e.g. one course) produces the same entries as a full run.
        docs_root = Path(self.settings.iu_docs_path).resolve()
        meta_root = docs_root if root == docs_root or docs_root in root.parents else root
        files = iter_source_files(
            root,
            include=self.settings.include_patterns(),
            exclude=self.settings.exclude_patterns() if meta_root == root else (),
            max_bytes=self.settings.max_file_mb * 1024 * 1024,
        )
        if meta_root != root:
            files = [
                f
                for f in files
                if not _matches(f.relative_to(meta_root).as_posix(), self.settings.exclude_patterns())
            ]
        documents: list[SourceDocument] = []
        for path in files:
            meta = infer_file_meta(meta_root, path)
            try:
                fingerprint = file_fingerprint(path)
            except OSError as exc:
                fingerprint = f"unreadable:{exc}"
            documents.append(
                SourceDocument(
                    source=f"{ORIGIN_ONEDRIVE}:{meta['rel_path']}",
                    fingerprint=fingerprint,
                    meta=meta,
                    path=path,
                )
            )
        return documents

    def ingest_folder(
        self,
        root: Path | None = None,
        *,
        force: bool = False,
        prune: bool = True,
        on_progress: ProgressCallback | None = None,
    ) -> IngestReport:
        documents = self.collect_folder(root)
        report = self.ingest(documents, force=force, on_progress=on_progress)
        if prune:
            report.removed = self.prune(ORIGIN_ONEDRIVE, {d.source for d in documents})
        return report
