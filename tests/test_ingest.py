from pathlib import Path

from iu_agent.rag.ingest import Ingestor, Manifest, infer_file_meta, iter_source_files


def _write_docs(root: Path) -> None:
    maths = root / "Semester_1" / "Advanced Maths"
    maths.mkdir(parents=True)
    (maths / "DLMDSAM01-01_notes.md").write_text(
        "# Eigenvalues\n\nAn eigenvalue lambda satisfies A v = lambda v for a matrix A and vector v.\n" * 5,
        encoding="utf-8",
    )
    deep = root / "Semester_2" / "Deep Learning"
    deep.mkdir(parents=True)
    (deep / "lecture.txt").write_text(
        "Neural networks learn weights with backpropagation and gradient descent.\n" * 5, encoding="utf-8"
    )
    bill = root / "Bill"
    bill.mkdir()
    (bill / "invoice.txt").write_text("Invoice amount 1234 euro", encoding="utf-8")


def test_iter_source_files_respects_excludes(settings):
    _write_docs(settings.iu_docs_path)
    files = iter_source_files(settings.iu_docs_path, exclude=settings.exclude_patterns())
    names = sorted(f.name for f in files)
    assert names == ["DLMDSAM01-01_notes.md", "lecture.txt"]


def test_infer_meta(settings):
    _write_docs(settings.iu_docs_path)
    path = settings.iu_docs_path / "Semester_1" / "Advanced Maths" / "DLMDSAM01-01_notes.md"
    meta = infer_file_meta(settings.iu_docs_path, path)
    assert meta["course"] == "Advanced Maths"
    assert meta["semester"] == "Semester_1"
    assert meta["course_code"] == "DLMDSAM01"
    assert meta["origin"] == "onedrive"


def test_incremental_folder_ingest_and_search(settings, store):
    _write_docs(settings.iu_docs_path)
    ingestor = Ingestor(settings, store)

    report = ingestor.ingest_folder()
    assert (report.added, report.updated, report.skipped, report.failed) == (2, 0, 0, [])
    assert store.count() == report.chunks > 0
    manifest = Manifest(settings.manifest_path)
    assert manifest.courses() == {"Advanced Maths": 1, "Deep Learning": 1}
    entry = manifest.get("onedrive:Semester_1/Advanced Maths/DLMDSAM01-01_notes.md")
    assert entry["course_code"] == "DLMDSAM01"

    # unchanged files are skipped
    report = ingestor.ingest_folder()
    assert (report.added, report.updated, report.skipped) == (0, 0, 2)

    # a changed file is re-indexed, a removed file is pruned
    lecture = settings.iu_docs_path / "Semester_2" / "Deep Learning" / "lecture.txt"
    lecture.write_text(
        "Convolutional neural networks use kernels and pooling layers.\n" * 5, encoding="utf-8"
    )
    (settings.iu_docs_path / "Semester_1" / "Advanced Maths" / "DLMDSAM01-01_notes.md").unlink()
    report = ingestor.ingest_folder()
    assert (report.added, report.updated, report.removed) == (0, 1, 1)
    assert ingestor.manifest.get("onedrive:Semester_1/Advanced Maths/DLMDSAM01-01_notes.md") is None

    hits = store.search("convolutional kernels pooling", k=2)
    assert hits and hits[0][0].metadata["course"] == "Deep Learning"
    assert "kernels" in hits[0][0].page_content
    assert store.search("eigenvalue", k=2, course="Advanced Maths") == []


def test_reset_clears_everything(settings, store):
    _write_docs(settings.iu_docs_path)
    ingestor = Ingestor(settings, store)
    ingestor.ingest_folder()
    ingestor.remove_all()
    assert store.count() == 0
    assert Manifest(settings.manifest_path).documents == {}


def test_identical_files_are_embedded_once(settings, store):
    root = settings.iu_docs_path
    (root / "Semester_3" / "Statistics").mkdir(parents=True)
    (root / "Semester_3" / "Statistics" / "My Library").mkdir()
    text = "Bayes theorem relates conditional probabilities of two events. " * 40
    (root / "Semester_3" / "Statistics" / "book.txt").write_text(text, encoding="utf-8")
    (root / "Semester_3" / "Statistics" / "My Library" / "book.txt").write_text(text, encoding="utf-8")
    ingestor = Ingestor(settings, store)
    report = ingestor.ingest_folder()
    assert (report.added, report.duplicates) == (1, 1)
    entries = ingestor.manifest.documents
    assert entries["onedrive:Semester_3/Statistics/My Library/book.txt"]["duplicate_of"] == (
        "onedrive:Semester_3/Statistics/book.txt"
    )
    assert ingestor.manifest.courses() == {"Statistics": 1}
    assert store.count() == entries["onedrive:Semester_3/Statistics/book.txt"]["chunks"]
    # removing the original drops the duplicate entry so the copy is embedded next time
    (root / "Semester_3" / "Statistics" / "book.txt").unlink()
    report = ingestor.ingest_folder()
    assert report.removed == 1 and report.added == 1 and report.duplicates == 0
    assert store.count() > 0


def test_oversized_documents_are_truncated(settings, store):
    settings.max_chunks_per_document = 2
    (settings.iu_docs_path / "big.txt").write_text(
        "Gradient descent updates the weights. " * 400, encoding="utf-8"
    )
    ingestor = Ingestor(settings, store)
    seen = []
    report = ingestor.ingest_folder(on_progress=lambda done, total, name: seen.append(name))
    assert report.truncated == 1
    assert ingestor.manifest.get("onedrive:big.txt")["chunks"] == 2
    assert ingestor.manifest.get("onedrive:big.txt")["truncated"] is True
    assert seen[0] == "big.txt"
