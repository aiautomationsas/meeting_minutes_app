from typing import Literal
from company_research_interrupt.state.types import ResearchState

def should_continue(state: ResearchState) -> Literal["tools", "human_review", "generate_report"]:
    """Decide workflow path."""
    messages = state.get('messages', [])
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    
    if state.get('awaiting_review', True):
        return "human_review"
    
    if not state.get('awaiting_review', True) or state.get('research_count', 0) >= 3:
        return "generate_report"
    
    return "research" 