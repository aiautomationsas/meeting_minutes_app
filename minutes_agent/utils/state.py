from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing import TypedDict, Annotated, Sequence, List, Optional

class Attendee(TypedDict):
    name: str
    position: str
    role: str

class Task(TypedDict):
    responsible: str
    date: str
    description: str

class MinutesContent(TypedDict):
    title: str
    date: str
    attendees: List[Attendee]
    summary: str
    takeaways: List[str]
    conclusions: List[str]
    next_meeting: List[str]
    tasks: List[Task]
    message: Optional[str]

class MinutesGraphState(TypedDict):
    audioFile: Optional[str]
    transcript: str
    wordCount: int
    minutes: MinutesContent
    critique: str
    outputFormatMeeting: str
    approved: bool
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Valor por defecto para messages
def default_messages() -> List[BaseMessage]:
    return []