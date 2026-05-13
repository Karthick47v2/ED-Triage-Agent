"""Normalize LangChain messages before LangGraph checkpoint serialization."""

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage
from pydantic import BaseModel


def message_content_to_plain_text(content: Any) -> str:
    """Coerce ``BaseMessage.content`` to a plain string.

    Chat Completions usually return a ``str``; the Responses API (and some
    multimodal models) may return a list of parts such as
    ``{"type": "text", "text": "..."}``.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                # Responses API can include non-user-visible blocks (e.g. reasoning).
                # Only keep assistant-facing text blocks to avoid leaking internals
                # into the interview transcript/state.
                block_type = item.get("type")
                if block_type not in ("text", "output_text", None):
                    continue
                t = item.get("text")
                if not isinstance(t, str):
                    c = item.get("content")
                    t = c if isinstance(c, str) else None
                if isinstance(t, str):
                    parts.append(t)
            else:
                t = getattr(item, "text", None)
                if isinstance(t, str):
                    parts.append(t)
        return "".join(parts)
    return str(content)


def sanitize_ai_message_for_checkpoint(message: BaseMessage) -> BaseMessage:
    """Convert structured-output ``parsed`` payloads on AIMessage to plain dicts.

    OpenAI/LangChain store Pydantic results in ``additional_kwargs["parsed"]``. Nested
    BaseModel values can trigger Pydantic serializer warnings (and awkward msgpack
    edges) when the graph checkpointer serializes state.
    """
    if not isinstance(message, AIMessage):
        return message
    parsed = message.additional_kwargs.get("parsed")
    if isinstance(parsed, BaseModel):
        ak = dict(message.additional_kwargs)
        ak["parsed"] = parsed.model_dump(mode="json")
        return message.model_copy(update={"additional_kwargs": ak})
    return message
