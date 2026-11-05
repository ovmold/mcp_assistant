import importlib.util
import sqlite3
from pathlib import Path

from mcp_assistant import config as conf

module_path = Path(__file__).resolve().parent.parent / "evals" / "conf_evals.py"
spec = importlib.util.spec_from_file_location("conf_evals", module_path)
conf_evals = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conf_evals)


def test_init_eval_db_creates_parent_directory(tmp_path, monkeypatch):
    db_path = tmp_path / "nested" / "evals.db"
    monkeypatch.setattr(conf, "DB_PATH", db_path)

    conf_evals.init_eval_db()

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
    assert count == 10
