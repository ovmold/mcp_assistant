from dataclasses import dataclass, field
from typing import Literal, get_args

TicketStatus = Literal["open", "pending", "resolved", "closed"]
TicketPriority = Literal["low", "medium", "high", "critical"]
TICKET_STATUSES = get_args(TicketStatus)
TICKET_PRIORITIES = get_args(TicketPriority)

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
    status: TicketStatus
    priority: TicketPriority
    client: str
    created_at: str
    updated_at: str
    tags: str
    issue_type: str
    messages: list[Message] = field(default_factory=list)
    logs: list[LogEntry] = field(default_factory=list)


@dataclass
class CommentAddResult:
    ticket_id: int
    comment_id: int
    added: bool


@dataclass
class StatusUpdateResult:
    ticket_id: int
    old_status: TicketStatus
    new_status: TicketStatus
    updated: bool
    comment_id: int
