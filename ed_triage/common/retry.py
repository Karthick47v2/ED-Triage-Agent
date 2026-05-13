"""Shared retry helper for structured-output LangChain chains."""
import logging
from typing import Any, TypeVar

T = TypeVar("T")
logger = logging.getLogger(__name__)

MAX_STRUCTURED_OUTPUT_RETRIES = 2


def invoke_with_retry(chain, inputs: dict, label: str = "Agent") -> Any:
    """Invoke a structured-output chain, retrying on None output.

    Args:
        chain: A compiled LangChain runnable returning structured output.
        inputs: Input dict for chain.invoke().
        label: Agent label for log messages and error text.

    Returns:
        The first non-None structured output.

    Raises:
        ValueError: If all retries return None.
    """
    for attempt in range(MAX_STRUCTURED_OUTPUT_RETRIES + 1):
        result = chain.invoke(inputs)
        if result is not None:
            return result
        logger.warning(
            "%s structured output missing on attempt %d/%d",
            label, attempt + 1, MAX_STRUCTURED_OUTPUT_RETRIES + 1,
        )
    raise ValueError(
        f"{label} returned no structured output after {MAX_STRUCTURED_OUTPUT_RETRIES + 1} "
        "attempts. Check model deployment/config and prompt compatibility."
    )
