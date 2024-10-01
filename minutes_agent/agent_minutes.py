import sys
sys.path.append('/home/santiago/ai_automation/langgraph_cloud/meeting_minutes_app')
import json
from langgraph.graph import StateGraph, END, START
#from langgraph.checkpoint.memory import MemorySaver
from minutes_agent.utils.nodes import read_transcript, create_minutes, create_critique, human_critique, human_approved, output_meeting
from minutes_agent.utils.state import MinutesGraphState



def should_continue(state: MinutesGraphState) -> str:
    critique = state.get('critique', '')
    if critique == "":
        return "human_approved"
    return "create_minutes"

# Crear el grafo
workflow = (
    StateGraph(MinutesGraphState)
    # Begin with a call LLM
    .add_node("read_transcript", read_transcript) # call model and include read_transcript into create_minutes
    .add_node("create_minutes", create_minutes)
    .add_node("create_critique", create_critique)
    .add_node("human_critique", human_critique)
    .add_node("human_approved", human_approved)
    .add_node("output_meeting", output_meeting)
    .add_edge(START, "read_transcript")
    .add_edge("read_transcript", "create_minutes")
    .add_edge("create_minutes", "create_critique")
    .add_edge("create_critique", "human_critique")
    .add_conditional_edges("human_critique", should_continue)
    .add_edge("human_approved", "output_meeting")
    .add_edge("output_meeting", END)
)

#checkpointer = MemorySaver()
graph =  workflow.compile()
