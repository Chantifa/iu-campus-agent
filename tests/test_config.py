from pathlib import Path

from iu_agent.config import find_env_file, load_settings, project_root

NEWLINE = chr(10)


def test_project_root_points_at_the_repo():
    root = project_root()
    assert root is not None and (root / "pyproject.toml").exists()


def test_env_file_in_cwd_wins(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("IU_AGENT_ENV_FILE", raising=False)
    (tmp_path / ".env").write_text("DATA_DIR=here" + NEWLINE, encoding="utf-8")
    assert find_env_file() == tmp_path / ".env"
    settings = load_settings()
    assert settings.data_dir == Path("here")


def test_project_env_and_relative_data_dir_from_elsewhere(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # no .env here
    for name in ("DATA_DIR", "QDRANT_URL", "IU_AGENT_ENV_FILE"):
        monkeypatch.delenv(name, raising=False)
    root = project_root()
    project_env = root / ".env"
    assert find_env_file() in (project_env, None)
    if project_env.is_file():
        settings = load_settings()
        assert settings.data_dir.is_absolute()
        assert settings.data_dir.parent == root or root in settings.data_dir.parents


def test_env_file_override_disables_lookup(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("DATA_DIR=ignored" + NEWLINE, encoding="utf-8")
    monkeypatch.setenv("IU_AGENT_ENV_FILE", "")
    assert find_env_file() is None
    monkeypatch.setenv("IU_AGENT_ENV_FILE", str(tmp_path / ".env"))
    assert find_env_file() == tmp_path / ".env"
