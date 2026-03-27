"""Message translation layer between A2A protocol and LangChain messages."""

from __future__ import annotations

import uuid

from a2a.types import Message, Part, Role, TextPart
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage


def a2a_to_langchain(a2a_msg: Message) -> BaseMessage:
    """Convert an A2A Message to a LangChain BaseMessage."""
    text_parts: list[str] = []
    for part in a2a_msg.parts:
        if hasattr(part.root, "text"):
            text_parts.append(part.root.text)
    content = "\n".join(text_parts) if text_parts else ""

    if a2a_msg.role == Role.user:
        return HumanMessage(content=content)
    return AIMessage(content=content)


def langchain_to_a2a(lc_msg: BaseMessage) -> Message:
    """Convert a LangChain BaseMessage to an A2A Message."""
    role = Role.user if isinstance(lc_msg, HumanMessage) else Role.agent
    content = lc_msg.content if isinstance(lc_msg.content, str) else str(lc_msg.content)
    return Message(
        role=role,
        parts=[Part(root=TextPart(text=content))],
        message_id=str(uuid.uuid4()),
    )


def langchain_messages_to_a2a(messages: list[BaseMessage]) -> list[Message]:
    """Convert a list of LangChain messages to A2A Messages."""
    return [langchain_to_a2a(msg) for msg in messages]
