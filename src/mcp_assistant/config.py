"""Пути к данным и параметры, зависящие от переменных окружения."""

from pathlib import Path
import os

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent.parent
CLIENT_DIR = PACKAGE_DIR / "client"


def _resolve_data_paths() -> None:
    """Пересчитывает пути после загрузки переменных окружения."""
    global DATA_DIR, TRACE_DIR, KB_DIR, DB_PATH

    DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_DIR / "data")).resolve()
    TRACE_DIR = Path(os.environ.get("TRACE_DIR", PROJECT_DIR / "traces")).resolve()
    KB_DIR = Path(os.environ.get("KB_DIR", DATA_DIR / "kb")).resolve()
    DB_PATH = Path(os.environ.get("DB_PATH", DATA_DIR / "tickets.db")).resolve()


_resolve_data_paths()

# Хост и порт для Streamable HTTP
HOST = "127.0.0.1"
PORT = 8000

N_SEARCH_LIMIT = 100 # макс. кол-во тикетов для поиска
N_SEARCH_ARTICLES = 10 # макс. кол-во статей для поиска
THRESHOLD = 0.2 # порог релевантности для фильтрации при поиске статей


def load_env(path: str | Path | None = None) -> None:
    """Загружает .env и обновляет зависимые от окружения пути."""
    from dotenv import load_dotenv
    load_dotenv(path)
    _resolve_data_paths()


def ensure_dirs() -> None:
    """Создает каталоги данных. Вызывать явно при инициализации."""
    for d in (DATA_DIR, TRACE_DIR, KB_DIR):
        d.mkdir(parents=True, exist_ok=True)





