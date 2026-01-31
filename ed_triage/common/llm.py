"""LLM configuration for ED triage agents."""
import os
from typing import Optional
from langchain_openai import AzureChatOpenAI


def get_llm(
    deployment: Optional[str] = None,
    temperature: float = 0,
) -> AzureChatOpenAI:
    """LLM for CRA, PAA, TCA."""
    return AzureChatOpenAI(
        azure_deployment=deployment or os.environ.get("AZURE_OPENAI_GENERAL_DEPLOYMENT_NAME", "gpt-4.1-mini"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        temperature=temperature,
    )


def get_llm_iia(
    deployment: Optional[str] = None,
    temperature: float = 0,
    streaming: bool = True,
) -> AzureChatOpenAI:
    """LLM for IIA (patient-facing)."""
    return AzureChatOpenAI(
        azure_deployment=deployment or os.environ.get("AZURE_OPENAI_IIA_DEPLOYMENT_NAME", "gpt-4.1-nano"),
        api_version=os.environ.get("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        temperature=temperature,
        streaming=streaming,
    )
