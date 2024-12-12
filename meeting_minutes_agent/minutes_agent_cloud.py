"""
Uso para probar en local

import sys
import os

# Añadir el directorio padre al PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
import json
from typing import Annotated, List, Literal
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()

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
    feedback_response: Annotated[str, ..., "Response to the reviewer on the changes made. Do not include the meeting minutes, only with the response to the critique."]

llm = ChatAnthropic(
    model="claude-3-5-haiku-20241022", max_tokens=8000
)

async def generation_node(state: State) -> State:
    meeting_minutes_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "As an expert in minute meeting creation, you are a chatbot designed to facilitate the process of "
            "generating meeting minutes efficiently.\n" 
            "Ensure that your responses are structured, concise, and provide a comprehensive overview of the meeting proceedings for"
            "effective record-keeping and follow-up actions.\n" 
            "If the user provides critique, respond with a revised version of your previous attempts.\n"
            "Respond in Spanish.\n", 
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])

    generate = meeting_minutes_prompt | llm.with_structured_output(MeetingMinutes)
    result = await generate.ainvoke(state["messages"])
    result_str = json.dumps(result, ensure_ascii=False, indent=2)
    
    return {"messages": [AIMessage(content=result_str)]}

async def reflection_node(state: State) -> State:
    reflection_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting minutes creator. Generate critique and recommendations for the meeting minutes provided."
            "Respond only with the critique and recommendations, no other text."
            "If the meeting minutes provided is already perfect, just say so."
            "You must respect the structure of the meeting minutes provided. Do not add or remove any sections."
            "The meeting minutes provided is given in the first message of the user."
            "Respond in Spanish language",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])
    reflect = reflection_prompt | llm
    
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    translated = [state["messages"][0]] + [
        cls_map[msg.type](content=msg.content) for msg in state["messages"][1:]
    ]
    
    res = await reflect.ainvoke(translated)
    return {"messages": [HumanMessage(content=res.content)]}

async def human_critique_node(state: State) -> State:
    return {"messages": [HumanMessage(content="")]}

async def revision_minutes_node(state: State) -> State:
    revision_prompt = ChatPromptTemplate.from_messages([
        (
            "system", 
            "Revises the previous minutes considering the criticisms and comments received. "
            "Makes adjustments to address comments accurately and professionally. "
            "If you are asked to add information that is not included in the minutes, first review the meeting transcript for context. If not, include what the user is asking for without adding any context. "
            "Respond in Spanish language",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])

    revise = revision_prompt | llm.with_structured_output(MeetingMinutes)
    result = await revise.ainvoke(state["messages"])
    result_str = json.dumps(result, ensure_ascii=False, indent=2)
    
    return {"messages": [AIMessage(content=result_str)]} 

def should_continue_reflection(state: State):
    if len(state["messages"]) > 2:
        # End after 1 iteration
        return "human_critique"
    return "reflect"

def should_continue_revision(state: State):
    last_message = state["messages"][-1].content
    if last_message.strip() == "aprobado" or not last_message.strip():
        return END
    return "revision"

builder = StateGraph(State)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("human_critique", human_critique_node)
builder.add_node("revision", revision_minutes_node)

builder.add_edge(START, "generate")
builder.add_conditional_edges("generate", should_continue_reflection)
builder.add_edge("reflect", "generate")
builder.add_conditional_edges("human_critique", should_continue_revision)
graph = builder.compile()

