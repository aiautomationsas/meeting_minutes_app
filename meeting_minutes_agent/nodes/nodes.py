import json
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic import ChatAnthropic
from meeting_minutes_agent.state.types import State, MeetingMinutes

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
    return state

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