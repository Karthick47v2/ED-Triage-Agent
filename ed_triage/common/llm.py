"""LLM configuration for ED triage agents."""
from __future__ import annotations

import os
from typing import Any, Optional
from functools import lru_cache

from langchain_openai import AzureChatOpenAI


def _azure_responses_chat_kwargs() -> dict[str, Any]:
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION")
    return {
        "azure_endpoint": os.environ.get("AZURE_OPENAI_ENDPOINT"),
        "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
        "api_version": api_version,
        "use_responses_api": True,
    }


@lru_cache(maxsize=1)
def get_llm(deployment: Optional[str] = None) -> AzureChatOpenAI:
    """LLM for CRA, PAA, TCA."""
    return AzureChatOpenAI(
        azure_deployment=deployment
        or os.environ.get("AZURE_OPENAI_GENERAL_DEPLOYMENT_NAME"),
        temperature=0.7,
        **_azure_responses_chat_kwargs(),
    )


@lru_cache(maxsize=1)
def get_llm_iia(
    deployment: Optional[str] = None,
) -> AzureChatOpenAI:
    """LLM for IIA."""
    return AzureChatOpenAI(
        azure_deployment=deployment
        or os.environ.get("AZURE_OPENAI_IIA_DEPLOYMENT_NAME"),
        temperature=0.7,
        **_azure_responses_chat_kwargs(),
    )


@lru_cache(maxsize=1)
def get_llm_patient(deployment: Optional[str] = None) -> AzureChatOpenAI:
    """Eval patient simulator; same model profile as IIA."""
    return get_llm_iia(deployment=deployment)
