"""
Uso para probar en local
"""
import sys
import os
import json
# Añadir el directorio padre al PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Annotated, List
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
import asyncio
#import os
from meeting_minutes_agent.utils.tools import read_transcript  # Importación absoluta
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

llm = ChatOpenAI(
    model="gpt-3.5-turbo-16k", max_tokens=2000
)

async def generation_node(state: State) -> State:
    meeting_minutes_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "As an expert in minute meeting creation, you are a chatbot designed to facilitate the process of "
                "generating meeting minutes efficiently.\n" 
                "Ensure that your responses are structured, concise, and provide a comprehensive overview of the meeting proceedings for"
                "effective record-keeping and follow-up actions.\n" 
                " If the user provides critique, respond with a revised version of your previous attempts.\n"
                "Respond in Spanish.\n", 
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

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

    generate = meeting_minutes_prompt | llm.with_structured_output(MeetingMinutes)
    
    result = await generate.ainvoke({"messages": state["messages"]})
    
    # Convertir el resultado estructurado a una cadena JSON
    result_str = json.dumps(result, ensure_ascii=False, indent=2)
    
    # Crear un nuevo AIMessage con el contenido del resultado
    new_message = AIMessage(content=result_str)
    
    # Devolver el estado actual más el nuevo AIMessage
    return {"messages": [new_message]} 

async def reflection_node(state: State) -> State:
    reflection_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert meeting minutes creator. Generate critique and recommendations for the user's submission."
                "Respond only with the critique and recommendations, no other text."
                "The transcript is given in the first message of the user."
                "Respond in Spanish language",
            ),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    reflect = reflection_prompt | llm
    
    # Other messages we need to adjust
    cls_map = {"ai": HumanMessage, "human": AIMessage}
    # First message is the original user request. We hold it the same for all nodes
    translated = [state["messages"][0]] + [
        cls_map[msg.type](content=msg.content) for msg in state["messages"][1:]
    ]
    res = await reflect.ainvoke(translated)
    
    return {"messages": [HumanMessage(content=res.content)]}


async def human_approved_node(state: State) -> State:
    print("\nMinutas generadas:")
    minutes = json.loads(state["messages"][-2].content)
    print(json.dumps(minutes, ensure_ascii=False, indent=2))
    
    print("\nCrítica generada:")
    print(state["messages"][-1].content)
    
    user_input = input("\n¿Tienes algún comentario sobre esta crítica? (Si no, presiona Enter para aprobar las actas): ").strip()
    
    if user_input == "":
        # Update the last message to indicate approval
        state["messages"][-1] = HumanMessage(content="")
    else:
        # Update the last message with the user's comment
        state["messages"][-1] = HumanMessage(content=user_input)
    
    # Mostrar en consola los mensajes del state con su tipo
    print("\nMensajes actuales en el estado:")
    for message in state["messages"]:
        print(f"{type(message).__name__}: {message.content}")
    
    return state

builder = StateGraph(State)
builder.add_node("generate", generation_node)
builder.add_node("human_approved", human_approved_node)
builder.add_node("reflect", reflection_node)
builder.add_edge(START, "generate")
builder.add_edge("generate", "reflect")
builder.add_edge("reflect", "human_approved")

"""
def should_continue(state: State):
    if len(state["messages"]) > 2:  # Si ya hemos generado y reflejado una vez
        return END
    else:
        return "reflect"
"""


def should_continue(state: State):
    if state["messages"][-1].content == "":
        return END
    else:
        return "generate"

builder.add_conditional_edges("human_approved", should_continue)


memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

"""
config = {"configurable": {"thread_id": "61"}}

async def process_events():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    transcript_path = os.path.join(project_root, 'transcripts', 'transcript.txt')
    
    transcript_content = read_transcript(transcript_path)
    
    initial_state = {
        "messages": [HumanMessage(content=f"Please help me create the meeting minutes for this transcript:\n\n{transcript_content}.\n\nRespond in Spanish language")],
    }

    async for event in graph.astream(initial_state, config):
        if isinstance(event, dict):
            for key, value in event.items():
                if key == "generate":
                    print("\nGenerando nuevas minutas...")
                elif key == "reflect":
                    print("\nGenerando crítica...")
                elif key == "human_approved":
                    if value["messages"][-1].content == "":
                        print("\nActas aprobadas. Proceso finalizado.")
                        return
                    else:
                        print("\nComentarios recibidos. Generando nuevas minutas...")
        elif event == END:
            print("\nProceso finalizado.")
            return
        else:
            print(f"Evento no reconocido: {event}")
        print("---")

if __name__ == "__main__":
    asyncio.run(process_events())
"""
