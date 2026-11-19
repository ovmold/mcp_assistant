import sqlite3
import sys

from mcp_assistant import config as conf
from mcp_assistant import seed


def test_seed_recreates_database_and_kb(tmp_path, monkeypatch) -> None:
    """Проверяет полное пересоздание БД и базы знаний seed-скриптом."""
    data_dir = tmp_path / "data"
    kb_dir = data_dir / "kb"
    kb_dir.mkdir(parents=True)
    (kb_dir / "stale.md").write_text("Устаревшая статья", encoding="utf-8")

    monkeypatch.setattr(conf, "DATA_DIR", data_dir)
    monkeypatch.setattr(conf, "TRACE_DIR", tmp_path / "traces")
    monkeypatch.setattr(conf, "KB_DIR", kb_dir)
    monkeypatch.setattr(conf, "DB_PATH", data_dir / "tickets.db")
    monkeypatch.setattr(conf, "load_env", lambda path: None)
    monkeypatch.setattr(seed, "make_tickets", lambda count: [])
    monkeypatch.setattr(sys, "argv", ["seed"])

    seed.main()

    assert not (kb_dir / "stale.md").exists()
    assert (kb_dir / "payment-502.md").exists()
    with sqlite3.connect(conf.DB_PATH) as conn:
        ticket_count = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
    assert ticket_count == 1
