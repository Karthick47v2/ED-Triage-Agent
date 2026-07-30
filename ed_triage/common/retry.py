"""Shared retry helper for structured-output LangChain chains."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Protocol, TypeVar


logger = logging.getLogger(__name__)

MAX_STRUCTURED_OUTPUT_RETRIES = 2

T = TypeVar("T")


class Invokable(Protocol[T]):
    def invoke(self, inputs: Mapping[str, Any]) -> T | None: ...


def invoke_with_retry(
    chain: Invokable[T],
    inputs: Mapping[str, Any],
    label: str = "Agent",
) -> T:
    """Invoke a structured-output chain, retrying when output is None."""
    total_attempts = MAX_STRUCTURED_OUTPUT_RETRIES + 1

    for attempt in range(1, total_attempts + 1):
        result = chain.invoke(inputs)
        if result is not None:
            return result

        logger.warning(
            "%s structured output missing on attempt %d/%d",
            label,
            attempt,
            total_attempts,
        )

    raise ValueError(
        f"{label} returned no structured output after "
        f"{total_attempts} attempts. Check model deployment/configuration "
        "and prompt compatibility."
    )
