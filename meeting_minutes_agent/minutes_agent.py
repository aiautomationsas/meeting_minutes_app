"""
import sys


# Añadir el directorio padre al PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""
from typing import Annotated, List
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from meeting_minutes_agent.utils.writerAgent import generate_meeting_minutes
from meeting_minutes_agent.utils.reflectionAgent import generate_reflection
from dotenv import load_dotenv
from langchain.schema import SystemMessage
import asyncio
import os
    
load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

async def generation_node(state: State) -> State:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    transcript_path = os.path.join(project_root, 'transcripts', 'transcript.txt')
    
    ai_message = await generate_meeting_minutes(state["messages"], transcript_path)
    return {"messages": [ai_message]}

async def reflection_node(state: State) -> State:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    transcript_path = os.path.join(project_root, 'transcripts', 'transcript.txt')
    
    reflection = await generate_reflection(state["messages"], transcript_path)
    return {"messages": [reflection]}
"""
def should_continue(state: State):
    if state["messages"][-1] == "":
        return END
    else:
        return "reflect"
"""

def should_continue(state: State):
    if len(state["messages"]) > 2:  # Si ya hemos generado y reflejado una vez
        return END
    else:
        return "reflect"

builder = StateGraph(State)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_edge(START, "generate")
builder.add_conditional_edges("generate", should_continue)
builder.add_edge("reflect", "generate")
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

"""
config = {"configurable": {"thread_id": "1"}}

async def process_events():
    initial_state = {
        "messages": [
            SystemMessage(content="Eres un asistente especializado en generar actas de reuniones. Tu tarea es crear un acta detallada y profesional basada en la transcripción proporcionada o, en su ausencia, en el tema dado."),
        ],
    }

    async for event in graph.astream(initial_state, config):
        print(event)
        print("---")

if __name__ == "__main__":
    asyncio.run(process_events())
"""
