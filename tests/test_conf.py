from pathlib import Path

from mcp_assistant import config as conf


def test_load_env_updates_data_paths(tmp_path, monkeypatch):
    names = ("DATA_DIR", "TRACE_DIR", "KB_DIR", "DB_PATH")
    for name in names:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.setattr(conf, name, getattr(conf, name))

    paths = {
        "DATA_DIR": tmp_path / "custom-data",
        "TRACE_DIR": tmp_path / "custom-traces",
        "KB_DIR": tmp_path / "custom-kb",
        "DB_PATH": tmp_path / "custom-tickets.db",
    }
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(f"{name}={path.as_posix()}" for name, path in paths.items()),
        encoding="utf-8",
    )

    conf.load_env(env_path)

    for name, path in paths.items():
        assert getattr(conf, name) == Path(path).resolve()
