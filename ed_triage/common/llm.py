"""LLM configuration for ED triage agents."""

from __future__ import annotations

import os
from functools import cache
from typing import Any

from langchain_openai import AzureChatOpenAI


def required_env(name: str) -> str:
    """Return a required environment variable or raise a clear error."""
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Required environment variable is not set: {name}")
    return value


def _azure_chat_kwargs() -> dict[str, Any]:
    return {
        "azure_endpoint": required_env("AZURE_OPENAI_ENDPOINT"),
        "api_key": required_env("AZURE_OPENAI_API_KEY"),
        "api_version": required_env("AZURE_OPENAI_API_VERSION"),
    }


@cache
def _create_llm(deployment: str, temperature: float) -> AzureChatOpenAI:
    """Create and cache one client per (deployment, temperature)."""
    return AzureChatOpenAI(
        azure_deployment=deployment,
        temperature=temperature,
        **_azure_chat_kwargs(),
    )


def get_llm(deployment: str | None = None) -> AzureChatOpenAI:
    """Return the LLM used by CRA, PAA, and TCA."""
    resolved = deployment or required_env(
        "AZURE_OPENAI_GENERAL_DEPLOYMENT_NAME"
    )
    return _create_llm(resolved, 0.0)


def get_llm_iia(deployment: str | None = None) -> AzureChatOpenAI:
    """Return the LLM used by IIA."""
    resolved = deployment or required_env(
        "AZURE_OPENAI_IIA_DEPLOYMENT_NAME"
    )
    return _create_llm(resolved, 0.7)


def get_llm_patient(deployment: str | None = None) -> AzureChatOpenAI:
    """Return the IIA-profile LLM used by the evaluation patient."""
    return get_llm_iia(deployment)
