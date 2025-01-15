import json
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic import ChatAnthropic
from meeting_minutes_agent.state.types import State, MeetingMinutes, Keypoints
from dotenv import load_dotenv

load_dotenv()

llm = ChatAnthropic(
    model="claude-3-5-haiku-20241022", max_tokens=8000
)

async def keypoints_analysis_node(state: State) ->State:
    analysis_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Your task is to review the provided meeting notes and extract key takeaways. \n"
            "Don't invent information that is not in the transcript. \n"
            "Respond in Spanish."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])

    messages_dict = {"messages": state["messages"]}
    
    keypoints = analysis_prompt | llm.with_structured_output(Keypoints)
    result = await keypoints.ainvoke(messages_dict)
    result_str = json.dumps(result, ensure_ascii=False, indent=2)
    
    return {"messages": [AIMessage(content=result_str)]}

async def human_keypoints_node(state: State) -> State:
    return state

async def revise_keypoints_node(state: State) -> State:
    revision_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "Review the key points considering the user's comments. \n"
            "Adjusts key points to accurately reflect feedback received. \n"
            "Do not add information that is not in the original transcript unless the user indicates it. \n"
            "Respond in Spanish."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])

    revise = revision_prompt | llm.with_structured_output(Keypoints)
    result = await revise.ainvoke(state["messages"])
    result_str = json.dumps(result, ensure_ascii=False, indent=2)
    
    return {"messages": [AIMessage(content=result_str)]}

async def generation_node(state: State) -> State:
    
    meeting_minutes_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "As an expert in creating meeting minutes, generate a complete record based on:\n"
            "1. The transcript of the meeting provided by the user in the first message.\n"
            "2. The key points previously approved by the user.\n"
            "3. Action items must be fully aligned with key points.\n"  
            "4. The assigned_actions must be fully aligned with the key points.\n"
            "Do not add information that is not in the transcript.\n"
            "Respond in Spanish."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])

    messages_dict = {"messages": state["messages"]}
    
    generate = meeting_minutes_prompt | llm.with_structured_output(MeetingMinutes)
    result = await generate.ainvoke(messages_dict)
    result_str = json.dumps(result, ensure_ascii=False, indent=2)
    
    return {"messages": [AIMessage(content=result_str)]}

async def reflection_node(state: State) -> State:
    reflection_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an expert meeting minutes creator. Generate critique and recommendations for the meeting minutes provided.\n"
            "Respond only with the critique and recommendations, no other text.\n"
            "All key points must be included in the meeting minutes.\n"
            "If the meeting minutes provided is already perfect, just say so.\n"
            "You must respect the structure of the meeting minutes provided. Do not add or remove any sections.\n"
            "The meeting transcript provided is given in the first message of the user.\n"
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
    return state

async def revision_minutes_node(state: State) -> State:
    revision_prompt = ChatPromptTemplate.from_messages([
        (
            "system", 
            "Revises the previous minutes considering the criticisms and comments received.\n "
            "Makes adjustments to address comments accurately and professionally. \n"
            "If you are asked to add information that is not included in the minutes, first review the meeting transcript for context. If not, include what the user is asking for without adding any context. \n"
            "Respond in Spanish language",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ])

    revise = revision_prompt | llm.with_structured_output(MeetingMinutes)
    result = await revise.ainvoke(state["messages"])
    result_str = json.dumps(result, ensure_ascii=False, indent=2)
    
    return {"messages": [AIMessage(content=result_str)]}  