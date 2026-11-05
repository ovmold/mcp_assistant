# ----------------------------- Скрипт для создания таблиц в БД ------------------------------

SCHEMA = """
PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS messages;
DROP TABLE IF EXISTS logs;
DROP TABLE IF EXISTS tickets_fts;   
DROP TABLE IF EXISTS tickets;

CREATE TABLE tickets (
    id          INTEGER PRIMARY KEY,
    subject     TEXT NOT NULL,
    body        TEXT NOT NULL,
    status      TEXT NOT NULL,
    priority    TEXT NOT NULL,
    client      TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    tags        TEXT,
    issue_type  TEXT
);

CREATE TABLE messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id   INTEGER NOT NULL REFERENCES tickets(id),
    author      TEXT NOT NULL,
    text        TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE TABLE logs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id   INTEGER NOT NULL REFERENCES tickets(id),
    level       TEXT NOT NULL,
    message     TEXT NOT NULL,
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
CREATE INDEX IF NOT EXISTS idx_tickets_priority ON tickets(priority);
CREATE INDEX IF NOT EXISTS idx_tickets_client ON tickets(client);
CREATE INDEX IF NOT EXISTS idx_tickets_created ON tickets(created_at);
CREATE INDEX IF NOT EXISTS idx_messages_ticket ON messages(ticket_id, created_at, id);
CREATE INDEX IF NOT EXISTS idx_logs_ticket ON logs(ticket_id, created_at, id);

-- Создает FTS5-таблицу 
CREATE VIRTUAL TABLE tickets_fts USING fts5(
    subject,
    body,
    content='tickets',
    content_rowid='id',
    tokenize='unicode61'
);

-- Триггеры для синхронизации
CREATE TRIGGER tickets_ai AFTER INSERT ON tickets BEGIN
    INSERT INTO tickets_fts(rowid, subject, body) VALUES (new.id, new.subject, new.body);
END;

CREATE TRIGGER tickets_au AFTER UPDATE ON tickets BEGIN
    INSERT INTO tickets_fts(tickets_fts, rowid, subject, body) VALUES ('delete', old.id, old.subject, old.body);
    INSERT INTO tickets_fts(rowid, subject, body) VALUES (new.id, new.subject, new.body);
END;

CREATE TRIGGER tickets_ad AFTER DELETE ON tickets BEGIN
    INSERT INTO tickets_fts(tickets_fts, rowid, subject, body) VALUES ('delete', old.id, old.subject, old.body);
END;
"""