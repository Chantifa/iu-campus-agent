from iu_agent.rag.chunking import build_header, chunk_sections
from iu_agent.rag.loaders import Section


def _chunks(source="onedrive:x.pdf"):
    sections = [Section("lorem ipsum dolor sit amet " * 200, {"page": 3, "pages": 10})]
    return chunk_sections(
        sections,
        source=source,
        base_meta={"course": "Advanced Maths", "file_name": "x.pdf", "origin": "onedrive"},
        chunk_size=500,
        chunk_overlap=50,
    )


def test_chunks_are_deterministic_and_carry_headers():
    first, second = _chunks(), _chunks()
    assert len(first) > 1
    assert [d.metadata["id"] for d in first] == [d.metadata["id"] for d in second]
    assert len({d.metadata["id"] for d in first}) == len(first)
    assert first[0].page_content.startswith("[Course: Advanced Maths | File: x.pdf | Page 3]")
    assert first[0].metadata["page"] == 3
    assert first[1].metadata["chunk"] == 1


def test_different_sources_get_different_ids():
    assert _chunks("onedrive:a.pdf")[0].metadata["id"] != _chunks("onedrive:b.pdf")[0].metadata["id"]


def test_header_skips_missing_fields():
    assert build_header({"course": "X", "slide": 2}) == "[Course: X | Slide 2]"
    assert build_header({}) == ""
