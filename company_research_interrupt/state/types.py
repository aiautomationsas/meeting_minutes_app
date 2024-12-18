from typing import TypedDict, Dict, Union, Annotated, List
from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages

class ResearchState(TypedDict):
    report: str
    documents: Dict[str, Dict[Union[str, int], Union[str, float]]]
    messages: Annotated[list[AnyMessage], add_messages]
    research_count: int
    awaiting_review: bool
    research_complete: bool
    