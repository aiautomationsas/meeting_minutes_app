from typing import Annotated, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

class Attendee(TypedDict):
    name: Annotated[str, ..., "Participant's full name"]
    position: Annotated[str, ..., "Professional title or role within the organization"]
    role: Annotated[str, ..., "Meeting participation role (e.g., Chair, Secretary, Stakeholder)"]

class Action(TypedDict):
    owner: Annotated[str, ..., "Action item owner/delegate"]
    due_date: Annotated[str, ..., "Target completion date"]
    description: Annotated[str, ..., "Detailed action item description and expected outcomes"]

class MeetingMinutes(TypedDict):
    title: Annotated[str, ..., "Official meeting title or agenda topic"]
    date: Annotated[str, ..., "Meeting date and time"]
    attendees: Annotated[List[Attendee], ..., "List of participants and their roles"]
    summary: Annotated[str, ..., "Executive summary highlighting key discussions and decisions"]
    key_points: Annotated[List[str], ..., "Strategic points and major discussion outcomes"]
    action_items: Annotated[List[str], ..., "Follow-up actions and agreed-upon decisions"]
    follow_up: Annotated[List[str], ..., "Next steps and agenda items for subsequent meeting"]
    assigned_actions: Annotated[List[Action], ..., "Detailed action items with ownership and deadlines"]
    feedback_response: Annotated[str, ..., "Response to the reviewer on the changes made"]