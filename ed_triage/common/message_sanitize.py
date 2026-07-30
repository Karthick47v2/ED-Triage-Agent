"""Normalize LangChain messages before LangGraph checkpoint serialization."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel


def message_content_to_plain_text(content: Any) -> str:
    """Coerce AIMessage-style content to a single plain-text string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
            else:
                text = getattr(block, "text", None)
                if isinstance(text, str):
                    parts.append(text)
        return "".join(parts)
    return str(content)


def sanitize_ai_message_for_checkpoint(
    message: BaseMessage,
) -> BaseMessage:
    """Convert an AIMessage's parsed Pydantic payload to a plain dictionary."""
    if not isinstance(message, AIMessage):
        return message

    parsed = message.additional_kwargs.get("parsed")
    if not isinstance(parsed, BaseModel):
        return message

    additional_kwargs = dict(message.additional_kwargs)
    additional_kwargs["parsed"] = parsed.model_dump(mode="json")

    return message.model_copy(
        update={"additional_kwargs": additional_kwargs}
    )
