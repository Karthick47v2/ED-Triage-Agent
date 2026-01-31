from typing import List, Annotated, Optional
from typing_extensions import TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from ed_triage.iia.schema import IntakeSummary
from ed_triage.cra.schema import CRAResult
from ed_triage.paa.schema import PriorityAssessment

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    intake_data: Optional[IntakeSummary]
    cra_result: Optional[CRAResult]
    paa_result: Optional[PriorityAssessment]
