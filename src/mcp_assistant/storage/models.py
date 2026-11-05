from dataclasses import dataclass, field

# ----------------------------- Класс LogEntry ------------------------------
@dataclass
class LogEntry():
    id: int
    level: str
    message: str
    created_at: str

# ----------------------------- Класс Message ------------------------------
@dataclass
class Message():
    id: int
    author: str
    text: str
    created_at: str

# ----------------------------- Класс Ticket ------------------------------
@dataclass
class Ticket:
    id: int
    subject: str
    body: str
    status: str
    priority: str
    client: str
    created_at: str
    updated_at: str
    tags: str
    issue_type: str
    messages: list[Message] = field(default_factory=list)
    logs: list[LogEntry] = field(default_factory=list)
