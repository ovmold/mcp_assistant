"""
MCP-сервер `knowledge` (транспорт Streamable HTTP)

Сервер работает поверх папки со статьями в Markdown. При запуске загружает
локальную модель эмбеддингов и индексирует статьи.

- **Tools:** `search_kb(query, top_k=3)` — гибридный поиск по статьям,
  объединяющий BM25 и локальные эмбеддинги без внешнего векторного сервиса;
  `get_article(article_type)` — текст статьи.
- **Resources:** `kb://{article_type}` — текст статьи.
"""

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from sentence_transformers import SentenceTransformer
import math
import numpy as np
import re

from mcp_assistant import config as conf

# Создает MCP сервер
mcp = MCPServer("knowledge base")

# Глобальный кэш модели и эмбеддингов
_EMBEDDER = None
_KB_EMBEDDINGS = None
_KB_FILENAMES = None
_KB_ARTICLES = None
_KB_BM25 = None

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


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


class _BM25Index:
    """Небольшой BM25-индекс для лексических совпадений без внешних сервисов."""

    def __init__(self, documents: list[str]) -> None:
        self._documents = [_tokenize(document) for document in documents]
        self._doc_lengths = np.array([len(document) for document in self._documents])
        self._avg_length = (
            float(self._doc_lengths.mean()) if self._doc_lengths.size else 0.0
        )
        self._doc_freqs = {}
        for document in self._documents:
            for token in set(document):
                self._doc_freqs[token] = self._doc_freqs.get(token, 0) + 1

    def scores(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self._documents), dtype=float)
        if not self._documents or self._avg_length == 0:
            return scores

        for token in _tokenize(query):
            doc_freq = self._doc_freqs.get(token, 0)
            if not doc_freq:
                continue
            inverse_doc_freq = math.log(
                1 + (len(self._documents) - doc_freq + 0.5) / (doc_freq + 0.5)
            )
            for index, document in enumerate(self._documents):
                frequency = document.count(token)
                if not frequency:
                    continue
                length_normalizer = 1.5 * (
                    1 - 0.75 + 0.75 * self._doc_lengths[index] / self._avg_length
                )
                scores[index] += inverse_doc_freq * (
                    frequency * 2.5 / (frequency + length_normalizer)
                )
        return scores


# ----------------------------- Tools ------------------------
def _init_kb() -> None:
    """Загружает модель и кодирует документы при старте или первом использовании."""
    global _EMBEDDER, _KB_EMBEDDINGS, _KB_FILENAMES, _KB_ARTICLES, _KB_BM25

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
        _KB_BM25 = None
        return

    _KB_FILENAMES = list(articles.keys())
    _KB_ARTICLES = articles

    texts = [articles[name] for name in _KB_FILENAMES]
    _KB_BM25 = _BM25Index(texts)
    # normalize_embeddings=True нужен для косинусной близости через скалярное произведение
    _KB_EMBEDDINGS = _EMBEDDER.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

@mcp.tool(
    annotations = ToolAnnotations(
        readOnlyHint=True
    )    
)
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

    # Кодирует запрос
    query_embedding = _EMBEDDER.encode(query, normalize_embeddings=True)

    semantic_scores = np.dot(_KB_EMBEDDINGS, query_embedding)
    lexical_scores = _KB_BM25.scores(query)
    max_lexical = float(lexical_scores.max()) if lexical_scores.size else 0.0
    if max_lexical > 0:
        scores = 0.3 * semantic_scores + 0.7 * lexical_scores / max_lexical
    else:
        scores = semantic_scores

    # Сортирует и берет top_k
    top_indices = np.argsort(scores)[::-1][:top_k]

    # Фильтрует нерелевантные (отрицательные или близкие к нулю)
    result = [
        _KB_FILENAMES[i]
        for i in top_indices
        if scores[i] > conf.THRESHOLD
    ]
    return result


def _read_article(article_type: str) -> str:
    """Возвращает статью; единая валидация имени предотвращает выход за KB_DIR."""
    if not re.fullmatch(r"[a-zA-Z0-9_\-]+", article_type):
        raise ValueError(f"Недопустимое имя статьи '{article_type}'.")
    return (conf.KB_DIR / f"{article_type}.md").read_text(encoding="utf-8")


@mcp.tool(
    annotations = ToolAnnotations(
        readOnlyHint=True
    )
)
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
        ToolError: Если имя недопустимо или статью не удалось прочитать.
    """
    try:
        return _read_article(article_type)
    except UnicodeDecodeError as e:
        raise ToolError(f"Файл статьи имеет некорректную кодировку: {e}") from e
    except ValueError as e:
        raise ToolError(str(e)) from e
    except FileNotFoundError as e:
        raise ToolError(f"Статья '{article_type}' не найдена.") from e
    except PermissionError as e:
        raise ToolError("Ошибка: нет прав на чтение файла статьи.") from e
    except Exception as e:
        raise ToolError(f"Ошибка чтения файла статьи: {e}") from e


# ----------------------------- Resources ------------------------
@mcp.resource("kb://{article_type}")
def get_article_resource(article_type: str) -> str:
    """Возвращает статью как MCP-ресурс `kb://{article_type}`."""
    try:
        return _read_article(article_type)
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(
            e.encoding,
            e.object,
            e.start,
            e.end,
            f"Файл статьи имеет некорректную кодировку: {e.reason}",
        ) from e
    except ValueError as e:
        raise ValueError(str(e)) from e
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Статья '{article_type}' не найдена.") from e
    except PermissionError as e:
        raise PermissionError("Ошибка: нет прав на чтение статьи.") from e
    except Exception as e:
        raise Exception(f"Ошибка чтения файла статьи: {e}") from e

def main() -> None:
    conf.load_env(conf.PROJECT_DIR / ".env")
    _init_kb()
    mcp.run(transport="streamable-http", host=conf.HOST, port=conf.PORT)


if __name__ == "__main__":
    main()