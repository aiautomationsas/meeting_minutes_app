from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from company_research_interrupt.state.types import ResearchState
from company_research_interrupt.nodes.nodes import call_model, tool_node, human_review_sources, write_report
from company_research_interrupt.nodes.conditions import should_continue

load_dotenv()

# Build graph
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("research", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("human_review", human_review_sources)
workflow.add_node("generate_report", write_report)

# Add edges
workflow.add_edge(START, "research")
workflow.add_conditional_edges(
    "research",
    should_continue,
    {
        "tools": "tools",
        "human_review": "human_review",
        "generate_report": "generate_report"
    }
)

workflow.add_edge("tools", "research")
workflow.add_edge("human_review", "generate_report")
workflow.add_edge("generate_report", END)

# Add memory
memory = MemorySaver()
graph = workflow.compile(checkpointer=memory)