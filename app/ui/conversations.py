"""In-session conversations (Phase 5). Nothing is persisted; there are no user accounts.

Each conversation keeps its own message list. Questions are answered independently by the backend (the
assistant has no conversational memory), so a conversation is a separate history, not a separate model
context.
"""

from __future__ import annotations

import itertools
import time
import uuid
from dataclasses import dataclass, field

TITLE_CHARS = 48


@dataclass
class Message:
    role: str  # "user" | "assistant"
    content: str
    status: str = "done"  # "pending" | "done" | "error" | "interrupted"
    response: dict | None = None  # the service response for assistant messages
    created: float = field(default_factory=time.time)


@dataclass
class Conversation:
    id: str
    title: str = "New conversation"
    messages: list[Message] = field(default_factory=list)
    created: float = field(default_factory=time.time)
    order: int = 0

    def add_question(self, text: str) -> Message:
        if not self.messages:
            self.title = text if len(text) <= TITLE_CHARS else text[:TITLE_CHARS - 1].rstrip() + "…"
        message = Message("user", text, status="pending")
        self.messages.append(message)
        return message

    def add_answer(self, question: Message, response: dict) -> Message:
        question.status = "done" if response.get("ok") else "error"
        message = Message("assistant", (response.get("result") or {}).get("answer") or response.get("error") or "",
                          status="done" if response.get("ok") else "error", response=response)
        self.messages.append(message)
        return message

    def mark_interrupted(self) -> int:
        """Questions left pending by a stopped run (e.g. the Stop button) are marked, not retried silently."""

        count = 0
        for message in self.messages:
            if message.role == "user" and message.status == "pending":
                message.status = "interrupted"
                count += 1
        return count

    def pending_question(self) -> Message | None:
        last = self.messages[-1] if self.messages else None
        return last if last is not None and last.role == "user" and last.status == "pending" else None

    def clear(self) -> None:
        self.messages.clear()
        self.title = "New conversation"


class ConversationStore:
    def __init__(self):
        self._items: dict[str, Conversation] = {}
        self._counter = itertools.count()
        self.active_id = self.new().id

    def new(self) -> Conversation:
        # Reuse an empty active conversation instead of piling up empty ones.
        current = self._items.get(getattr(self, "active_id", None) or "")
        if current is not None and not current.messages:
            return current
        conversation = Conversation(uuid.uuid4().hex[:12], order=next(self._counter))
        self._items[conversation.id] = conversation
        self.active_id = conversation.id
        return conversation

    @property
    def active(self) -> Conversation:
        return self._items[self.active_id]

    def select(self, conversation_id: str) -> Conversation:
        if conversation_id not in self._items:
            raise KeyError(conversation_id)
        self.active_id = conversation_id
        return self._items[conversation_id]

    def delete(self, conversation_id: str) -> None:
        self._items.pop(conversation_id, None)
        if not self._items:
            self.active_id = ""
            self.active_id = self.new().id
        elif self.active_id == conversation_id:
            self.active_id = self.list()[0].id

    def list(self) -> list[Conversation]:
        return sorted(self._items.values(), key=lambda c: -c.order)

    def __len__(self) -> int:
        return len(self._items)
