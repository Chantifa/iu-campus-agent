from typer.testing import CliRunner

from iu_agent import __version__
from iu_agent.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_status_and_models_without_providers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in (
        "ANTHROPIC_API_KEY",
        "MOONSHOT_API_KEY",
        "KIMI_API_KEY",
        "SWISSAI_API_KEY",
        "HF_TOKEN",
        "HUGGINGFACE_TOKEN",
        "OLLAMA_BASE_URL",
        "QDRANT_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_ENABLED", "false")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))

    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0, result.output
    assert "none configured" in result.output

    result = runner.invoke(app, ["models"])
    assert result.exit_code == 0
    assert "No provider configured" in result.output


def test_ingest_dry_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    docs = tmp_path / "docs" / "Semester_1" / "Maths"
    docs.mkdir(parents=True)
    (docs / "notes.md").write_text("# notes", encoding="utf-8")
    monkeypatch.setenv("IU_DOCS_PATH", str(tmp_path / "docs"))
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    result = runner.invoke(app, ["ingest", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "1 files would be indexed" in result.output
    assert "Maths" in result.output


def test_chat_refuses_without_terminal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    result = runner.invoke(app, ["chat"])
    assert result.exit_code == 2
    assert "interactive terminal" in result.output
