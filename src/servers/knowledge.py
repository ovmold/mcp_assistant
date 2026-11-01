"""
MCP-сервер `knowledge` (транспорт Streamable HTTP)

Сервер работает поверх папки со статьями в Markdown.

- **Tools:** `search_kb(query, top_k)` — поиск по статьям. Хватит BM25 или локальных эмбеддингов, внешний векторный сервис не нужен. `get_article(article_type)`.
- **Resources:** `kb://{article_type}` — текст статьи.
"""

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from sentence_transformers import SentenceTransformer
import numpy as np
import re

from scripts import conf

# Создает MCP сервер
mcp = MCPServer("knowledge base")

# Глобальный кэш модели и эмбеддингов
_EMBEDDER = None
_KB_EMBEDDINGS = None
_KB_FILENAMES = None
_KB_ARTICLES = None

def clean_markdown(text: str) -> str:
    """
    Убирает Markdown-разметку из текста, оставляя чистый текст.

    Удаляет:
    - код-блоки (```...```)
    - inline-код (`...`)
    - HTML-теги
    - изображения (![alt](url))
    - ссылки, оставляя текст ([text](url) → text)
    - заголовки (#, ##, ...)
    - жирный/курсив (**text**, *text*, __text__, _text_)
    - зачеркнутый (~~text~~)
    - горизонтальные линии (---, ***)
    - блочные цитаты (>)
    - маркеры списков (-, *, +, 1.)
    - таблицы (| ... | ... |)
    - лишние пробелы и пустые строки

    Args:
        text: Исходный Markdown-текст.

    Returns:
        Очищенный текст без разметки.
    """
    if not text:
        return ""

    # 1. Код-блоки ```...```
    text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)

    # 2. Inline-код `...`
    text = re.sub(r'`[^`]*`', ' ', text)

    # 3. HTML-теги <...>
    text = re.sub(r'<[^>]+>', ' ', text)

    # 4. Изображения ![alt](url) — удаляем полностью
    text = re.sub(r'!\[[^\]]*\]\([^\)]*\)', ' ', text)

    # 5. Ссылки [text](url) — оставляем только text
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)

    # 6. Заголовки (#, ##, ...) — убираем решетки в начале строки
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)

    # 7. Жирный/курсив (**text**, *text*, __text__, _text_)
    text = re.sub(r'(\*{1,3}|_{1,3})(.+?)\1', r'\2', text)

    # 8. Зачеркнутый ~~text~~
    text = re.sub(r'~~(.+?)~~', r'\1', text)

    # 9. Горизонтальные линии (---, ***, ___)
    text = re.sub(r'^[-*_]{3,}\s*$', '', text, flags=re.MULTILINE)

    # 10. Блочные цитаты (>)
    text = re.sub(r'^>\s?', '', text, flags=re.MULTILINE)

    # 11. Маркеры списков (-, *, +, 1.)
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # 12. Таблицы (| ... | ... |)
    text = re.sub(r'\|', ' ', text)
    text = re.sub(r'^[\s\-:|]+$', '', text, flags=re.MULTILINE)

    # 13. Множественные пробелы → один
    text = re.sub(r'[ \t]+', ' ', text)

    # 14. Множественные пустые строки → одна
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text.strip()

# ----------------------------- Tools ------------------------
def _init_kb():
    """Ленивая инициализация: загрузка модели и кодирование документов."""
    global _EMBEDDER, _KB_EMBEDDINGS, _KB_FILENAMES, _KB_ARTICLES

    if _EMBEDDER is not None:
        return

    # 1. Загружает модель 
    # "paraphrase-multilingual-MiniLM-L12-v2"
    _EMBEDDER = SentenceTransformer(     
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # 2. Загружает статьи
    articles = {}
    for path in sorted(conf.KB_DIR.glob("*.md")):
        try:
            content = path.read_text(encoding="utf-8")
            articles[path.stem] = clean_markdown(content)
        except Exception as e:
            raise Exception(f"Ошибка чтения {path.name}: {e}")

    # База знаний пуста
    if not articles:
        _KB_EMBEDDINGS = None
        return

    _KB_FILENAMES = list(articles.keys())
    _KB_ARTICLES = articles

    # 3. Кодирует все документы в эмбеддинги
    texts = [articles[name] for name in _KB_FILENAMES]
    # normalize_embeddings=True нужен для косинусной близости через скалярное произведение
    _KB_EMBEDDINGS = _EMBEDDER.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

@mcp.tool()
def search_kb(query: str, top_k: int = 3) -> list[str]: 
    """
    Ищет статьи в базе знаний по поисковому запросу.

    Args:
        query: Поисковый запрос (например, 'ошибка оплаты', 'сброс пароля').
        top_k: Максимальное количество возвращаемых статей (от 1 до 10).

    Returns:
        Список имен статей (без расширения .md), отсортированных по релевантности.
        Если ничего не найдено или произошла ошибка, возвращает пустой список.
    Raises:
        ToolError: Если ключевое слово или фраза для поиска (query) пустое.
    """

    # Ограничивает top_k разумными значениями
    top_k = max(1, min(top_k, conf.N_SEARCH_ARTICLES))

    # Пустой запрос
    if not query or not query.strip():
        raise ToolError(
            "Параметр query пуст. Пожалуйста, уточните у пользователя, "
            "что именно искать в статьях (например: 'ошибка 502', "
            "'сброс пароля', 'оплата')."
        )
       
    # Ленивая инициализация
    if _EMBEDDER is None:
        _init_kb()

    if _KB_EMBEDDINGS is None:
        return []

    # 1. Кодирует запрос
    query_embedding = _EMBEDDER.encode(query, normalize_embeddings=True)

    # 2. Считает косинусную близость через скалярное произведение
    # (т.к. векторы нормализованы, скалярное произведение = косинусная близость)
    scores = np.dot(_KB_EMBEDDINGS, query_embedding)

    # 3. Сортирует и берет top_k
    top_indices = np.argsort(scores)[::-1][:top_k]

    # 4. Фильтрует нерелевантные (отрицательные или близкие к нулю)
    result = [
        _KB_FILENAMES[i]
        for i in top_indices
        if scores[i] > conf.THRESHOLD
    ]
    return result


@mcp.tool()
def get_article(article_type: str) -> str: 
    """
    Возвращает содержимое статьи из базы знаний по ее типу.

    Инструмент читает Markdown-файл из директории базы знаний (KB_DIR)
    и возвращает его текст. Имя файла формируется как '<article_type>.md'.

    Args:
        article_type: Тип статьи (имя файла без расширения .md).
            Например: 'payment-502', 'password-reset-email', 'slow-calendar'.
            Допускаются только буквы, цифры, дефисы и подчеркивания.

    Returns:
        Текст статьи в формате Markdown.

    Raises:
        ToolError: 
            - Если имя статьи недопустимо (содержит запрещенные символы).
            - Если статья с указанным типом не найдена.
            - Если нет прав на чтение файла статьи.
            - Если файл статьи имеет некорректную кодировку.
            - При любых других ошибках чтения файла.
    """
    # Валидация: только буквы, цифры, дефисы, подчеркивания
    if not re.fullmatch(r"[a-zA-Z0-9_\-]+", article_type):
        raise ToolError(f"Недопустимое имя статьи '{article_type}'.")
    path = conf.KB_DIR / f"{article_type}.md"
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except FileNotFoundError:
            raise ToolError(f"Статья '{article_type}' не найдена.")
    except PermissionError:
        raise ToolError("Ошибка: нет прав на чтение файла статьи.")
    except UnicodeDecodeError as e:
        raise ToolError(
            f"Файл статьи имеет некорректную кодировку: {e}"
        ) from e
    except Exception as e:
        raise ToolError(f"Ошибка чтения файла статьи: {e}") 

# ----------------------------- Resources ------------------------
@mcp.resource("kb://{article_type}")
def get_article_resource(article_type: str) -> str:
    """
    Возвращает текст статьи из базы знаний как MCP-ресурс.

    URI: kb://{article_type}
    Например: kb://payment-502, kb://password-reset-email.

    Args:
        article_type: Тип статьи (имя файла без расширения .md).
            Допускаются только буквы, цифры, дефисы и подчеркивания.

    Returns:
        Текст статьи в формате Markdown.

    Raises:
        ValueError: Если имя статьи недопустимо.
        FileNotFoundError: Если статья с указанным типом не найдена.
        PermissionError: Если нет прав на чтение файла статьи.
        UnicodeDecodeError: Если файл статьи имеет некорректную кодировку.
        Exception: При любых других ошибках чтения файла.
    """
    # Валидация: только буквы, цифры, дефисы, подчеркивания
    if not re.fullmatch(r"[a-zA-Z0-9_\-]+", article_type):
        raise ValueError(f"Недопустимое имя статьи '{article_type}'")
    path = conf.KB_DIR / f"{article_type}.md"
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        raise FileNotFoundError(f"Статья '{article_type}' не найдена.")
    except PermissionError:
        raise PermissionError("Ошибка: нет прав на чтение статьи.")
    except UnicodeDecodeError:
        raise UnicodeDecodeError("Ошибка: файл статьи имеет некорректную кодировку.")
    except Exception as e:
        raise Exception(f"Ошибка чтения статьи: {e}")

if __name__ == "__main__":
    mcp.run(transport="streamable-http", host=conf.HOST, port=conf.PORT)