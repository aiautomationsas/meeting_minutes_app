from dotenv import load_dotenv
from langgraph.graph import END, StateGraph, START
from meeting_minutes_agent.state.types import State
from meeting_minutes_agent.nodes.nodes import (
    keypoints_analysis_node,
    generation_node,
    reflection_node,
    human_critique_node,
    revision_minutes_node,
    revise_keypoints_node,
    human_keypoints_node
)

load_dotenv()

def should_continue_keypoints_revision(state: State):
    last_message = state["messages"][-1].content
    if last_message.strip() == "aprobado" or not last_message.strip():
        return "generate"
    return "revise_keypoints"

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
builder.add_node("keypoints", keypoints_analysis_node)
builder.add_node("human_keypoints", human_keypoints_node)
builder.add_node("revise_keypoints", revise_keypoints_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("human_critique", human_critique_node)
builder.add_node("revision", revision_minutes_node)

builder.add_edge(START, "keypoints")
builder.add_edge("keypoints", "human_keypoints")
builder.add_conditional_edges("human_keypoints", should_continue_keypoints_revision)
builder.add_edge("revise_keypoints", "human_keypoints")
builder.add_conditional_edges("generate", should_continue_reflection)
builder.add_edge("reflect", "generate")
builder.add_conditional_edges("human_critique", should_continue_revision)
graph = builder.compile()

