from typing import TypedDict, Annotated, List, Literal, Dict
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, add_messages
from langchain_groq import ChatGroq
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()

# Debate Agent State
class DebateState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    topic: str
    perspectives: Dict[str, str]
    current_speaker: str
    debate_stage: Literal['topic_selection', 'perspective_assignment', 'opening_statements', 'debate', 'conclusion']
    debate_count: int
    debate_round: int

# Initialize different models for different roles with distinct personalities
moderator = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.3)
debater_a = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.2)  # More analytical
debater_b = ChatGroq(model="deepseek-r1-distill-llama-70b", temperature=0.7)  # More creative

def get_next_stage(current_stage: str) -> str:
    """Determine the next stage in the debate flow"""
    stages = {
        'topic_selection': 'perspective_assignment',
        'perspective_assignment': 'opening_statements',
        'opening_statements': 'debate',
        'debate': 'debate',  # Will be handled by should_continue_debate
    }
    return stages.get(current_stage, current_stage)

def debug_log(message: str, state: dict = None):
    """Print debug information"""
    print(f"\n🔍 DEBUG: {message}")
    if state:
        print(f"  Stage: {state.get('debate_stage')}")
        print(f"  Speaker: {state.get('current_speaker')}")
        print(f"  Topic: {state.get('topic')}")
        print(f"  Round: {state.get('debate_round', 1)}")
        print(f"  Count: {state.get('debate_count', 0)}")
        if state.get('perspectives'):
            print("  Perspectives:")
            print(f"    A: {state['perspectives'].get('a', 'Not defined')[:50]}...")
            print(f"    B: {state['perspectives'].get('b', 'Not defined')[:50]}...")
    print("-" * 40)

def topic_selection_node(state: DebateState):
    """Node to help select and refine the debate topic"""
    debug_log("Entering topic selection node", state)
    
    system_prompt = """You are an AI debate moderator named 'Moderador 🎭'.
    Help the user select and refine a debate topic.
    
    Ask clarifying questions to ensure the topic is:
    1. Clear and specific
    2. Has two or more valid perspectives
    3. Interesting and thought-provoking
    
    Once the user provides a topic, acknowledge it and ask if they would like to proceed with the debate."""
    
    messages = [
        SystemMessage(content=system_prompt)
    ] + state['messages']
    
    response = moderator.invoke(messages)
    
    # Move to next stage if we have a topic
    next_stage = get_next_stage('topic_selection') if state.get('topic') else 'topic_selection'
    
    result = {
        "messages": [response],
        "debate_stage": next_stage,
        "current_speaker": "Moderador 🎭",
        "topic": state.get('topic', '')
    }
    
    debug_log("Exiting topic selection node", result)
    return result

def perspective_assignment_node(state: DebateState):
    """Assign perspectives to AI debaters"""
    debug_log("Entering perspective assignment node", state)
    
    system_prompt = f"""You are the debate moderator 'Moderador 🎭'.
    For the topic: {state['topic']}
    
    Assign two distinct and well-reasoned perspectives for our debaters:
    - Debater A 🔵: A more analytical and evidence-based approach
    - Debater B 🔴: A more innovative and challenging perspective
    
    Present the perspectives clearly and ask the user if they agree with these viewpoints.
    
    Format your response as:
    [Perspective A]
    First perspective here...
    
    [Perspective B]
    Second perspective here...
    
    Do you agree with these perspectives for the debate?"""
    
    messages = [
        SystemMessage(content=system_prompt)
    ] + state['messages']
    
    response = moderator.invoke(messages)
    
    # Extract perspectives from the response
    content = response.content
    try:
        perspective_a = content.split("[Perspective A]")[1].split("[Perspective B]")[0].strip()
        perspective_b = content.split("[Perspective B]")[1].split("Do you agree")[0].strip()
        perspectives = {"a": perspective_a, "b": perspective_b}
        debug_log("Successfully extracted perspectives")
    except:
        debug_log("Failed to extract perspectives, using defaults")
        perspectives = state.get('perspectives', {"a": "Not defined", "b": "Not defined"})
    
    # Move to next stage if user agrees
    last_msg = state['messages'][-1].content.lower() if state['messages'] else ""
    should_proceed = any(word in last_msg for word in ['sí', 'si', 'yes', 'ok', 'procede', 'adelante'])
    next_stage = get_next_stage('perspective_assignment') if should_proceed else 'perspective_assignment'
    
    result = {
        "messages": [response],
        "debate_stage": next_stage,
        "current_speaker": "Debater A 🔵" if next_stage == "opening_statements" else "Moderador 🎭",
        "perspectives": perspectives,
        "topic": state['topic']
    }
    
    debug_log("Exiting perspective assignment node", result)
    return result

def opening_statements_node(state: DebateState):
    """Generate opening statements for each perspective"""
    debug_log("Entering opening statements node", state)
    
    current_speaker = state.get('current_speaker', 'Debater A 🔵')
    debug_log(f"Current speaker: {current_speaker}")
    
    if current_speaker == 'Debater A 🔵':
        model = debater_a
        system_prompt = f"""You are Debater A 🔵, known for analytical and evidence-based arguments.
        Topic: '{state['topic']}'
        Your perspective: {state['perspectives'].get('a', 'Not defined')}
        
        Make a strong, well-reasoned opening argument that:
        1. Clearly states your position
        2. Provides concrete evidence and data to support your view
        3. Sets up your key points for the debate
        
        Remember: You are the analytical, evidence-based debater. Use facts, statistics, and logical reasoning."""
    else:
        model = debater_b
        system_prompt = f"""You are Debater B 🔴, known for innovative and challenging perspectives.
        Topic: '{state['topic']}'
        Your perspective: {state['perspectives'].get('b', 'Not defined')}
        
        Make a compelling opening argument that:
        1. Challenges conventional thinking
        2. Introduces fresh perspectives and innovative ideas
        3. Sets up your key points for the debate
        
        Remember: You are the innovative, challenging debater. Think outside the box and challenge established views."""
    
    messages = [
        SystemMessage(content=system_prompt)
    ] + state['messages']
    
    response = model.invoke(messages)
    
    next_speaker = 'Debater B 🔴' if current_speaker == 'Debater A 🔵' else 'Debater A 🔵'
    next_stage = get_next_stage('opening_statements') if next_speaker == 'Debater A 🔵' else 'opening_statements'
    
    result = {
        "messages": [response],
        "debate_stage": next_stage,
        "current_speaker": next_speaker,
        "debate_count": 0,
        "debate_round": 1,
        "perspectives": state['perspectives'],
        "topic": state['topic']
    }
    
    debug_log("Exiting opening statements node", result)
    return result

def debate_node(state: DebateState):
    """Conduct the debate with back-and-forth arguments"""
    debug_log("Entering debate node", state)
    
    current_speaker = state.get('current_speaker', 'Debater A 🔵')
    debate_round = state.get('debate_round', 1)
    
    debug_log(f"Current speaker: {current_speaker}, Round: {debate_round}")
    
    if current_speaker == 'Debater A 🔵':
        model = debater_a
        system_prompt = f"""You are Debater A 🔵 in round {debate_round}.
        Topic: '{state['topic']}'
        Your perspective: {state['perspectives'].get('a', 'Not defined')}
        
        Address the previous arguments and present your case:
        1. Use facts and evidence to support your points
        2. Present new data and analysis
        3. Maintain a professional, evidence-based tone
        
        Remember: You are the analytical debater. Focus on logic and evidence."""
    else:
        model = debater_b
        system_prompt = f"""You are Debater B 🔴 in round {debate_round}.
        Topic: '{state['topic']}'
        Your perspective: {state['perspectives'].get('b', 'Not defined')}
        
        Challenge the previous arguments:
        1. Introduce innovative viewpoints
        2. Question underlying assumptions
        3. Propose creative solutions
        
        Remember: You are the innovative debater. Think differently and challenge conventions."""
    
    messages = [
        SystemMessage(content=system_prompt)
    ] + state['messages']
    
    response = model.invoke(messages)
    
    next_speaker = 'Debater B 🔴' if current_speaker == 'Debater A 🔵' else 'Debater A 🔵'
    
    result = {
        "messages": [response],
        "debate_stage": "debate",
        "current_speaker": next_speaker,
        "debate_count": state['debate_count'] + (1 if next_speaker == 'Debater A 🔵' else 0),
        "debate_round": debate_round + (1 if next_speaker == 'Debater A 🔵' else 0),
        "perspectives": state['perspectives'],
        "topic": state['topic']
    }
    
    debug_log("Exiting debate node", result)
    return result

def conclusion_node(state: DebateState):
    """Summarize the debate and provide insights"""
    system_prompt = f"""You are the Moderador 🎭 concluding the debate on '{state['topic']}'.
    
    Provide a comprehensive summary that:
    1. Highlights the strongest arguments from both debaters
    2. Identifies areas of agreement and disagreement
    3. Notes particularly innovative or well-supported points
    4. Encourages the audience to reflect on both perspectives
    
    Remember to thank both debaters and the audience for their participation."""
    
    messages = [
        SystemMessage(content=system_prompt)
    ] + state['messages']
    
    response = moderator.invoke(messages)
    
    return {
        "messages": [response],
        "debate_stage": "conclusion",
        "current_speaker": "Moderador 🎭"
    }

def should_continue_debate(state: DebateState) -> str:
    """Determine whether to continue the debate or conclude"""
    debug_log("Checking if debate should continue", state)
    if state['debate_count'] >= 3:  # Allow 3 rounds of debate
        debug_log("Debate should conclude")
        return "conclusion"
    debug_log("Debate should continue")
    return "debate"

# Create the debate workflow
workflow = StateGraph(DebateState)

# Add nodes
workflow.add_node("topic_selection", topic_selection_node)
workflow.add_node("perspective_assignment", perspective_assignment_node)
workflow.add_node("opening_statements", opening_statements_node)
workflow.add_node("debate", debate_node)
workflow.add_node("conclusion", conclusion_node)

# Set entry point
workflow.set_entry_point("topic_selection")

# Add edges
workflow.add_edge("topic_selection", "perspective_assignment")
workflow.add_edge("perspective_assignment", "opening_statements")
workflow.add_edge("opening_statements", "debate")
workflow.add_conditional_edges("debate", should_continue_debate, {
    "debate": "debate",
    "conclusion": "conclusion"
})
workflow.add_edge("conclusion", END)

# Compile the graph
graph = workflow.compile() 