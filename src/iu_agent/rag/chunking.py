"""Split sections into overlapping chunks with deterministic ids and a context header.

The header (``[Course: ... | File: ... | Page 3]``) is embedded together with the chunk text so
that retrieval works for questions like "what does the Advanced Maths course book say about ...",
and so the model can cite the source precisely.
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from iu_agent.rag.loaders import Section

NAMESPACE = uuid.UUID("3b1a5d6e-8c7f-4a2b-9d1e-0f6c5b4a3d2e")

_HEADER_FIELDS: tuple[tuple[str, str], ...] = (
    ("course", "Course"),
    ("course_code", "Code"),
    ("semester", "Semester"),
    ("module", "Module"),
    ("section", "Section"),
    ("file_name", "File"),
    ("page", "Page"),
    ("slide", "Slide"),
    ("sheet", "Sheet"),
)

_LOCATOR_KEYS = ("page", "slide", "sheet", "cell", "part")


def section_key(meta: dict[str, Any]) -> str:
    for key in _LOCATOR_KEYS:
        if key in meta:
            return f"{key}={meta[key]}"
    return ""


def chunk_id(source: str, locator: str, index: int) -> str:
    return str(uuid.uuid5(NAMESPACE, f"{source}|{locator}|{index}"))


def build_header(meta: dict[str, Any]) -> str:
    parts = []
    for key, label in _HEADER_FIELDS:
        value = meta.get(key)
        if value in (None, ""):
            continue
        if key in ("page", "slide", "sheet"):
            parts.append(f"{label} {value}")
        else:
            parts.append(f"{label}: {value}")
    return "[" + " | ".join(parts) + "]" if parts else ""


def make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
        length_function=len,
    )


def chunk_sections(
    sections: list[Section],
    *,
    source: str,
    base_meta: dict[str, Any],
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> list[Document]:
    splitter = make_splitter(chunk_size, chunk_overlap)
    documents: list[Document] = []
    for section in sections:
        locator = section_key(section.meta)
        for index, piece in enumerate(splitter.split_text(section.text)):
            piece = piece.strip()
            if not piece:
                continue
            meta: dict[str, Any] = {**base_meta, **section.meta, "source": source, "chunk": index}
            meta["id"] = chunk_id(source, locator, index)
            header = build_header(meta)
            content = f"{header}\n{piece}" if header else piece
            documents.append(Document(page_content=content, metadata=meta))
    return documents
