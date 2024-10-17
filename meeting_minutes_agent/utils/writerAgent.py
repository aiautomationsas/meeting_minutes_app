from typing import List, Optional
from typing_extensions import Annotated, TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from pydantic import BaseModel, Field
import os
from .tools import tools


# Nuevo prompt para actas de reunión
meeting_minutes_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un asistente experto en crear actas de reunión detalladas y estructuradas. "
            "Tienes acceso a una herramienta llamada TranscriptReader que puede leer el contenido de un archivo de transcript. "
            "Tu tarea es generar un acta completa basada en la información proporcionada por el usuario y el contenido del transcript. "
            "Asegúrate de incluir todos los elementos requeridos en la estructura MeetingMinutes. "
            "Si el usuario proporciona críticas o solicita cambios, responde con una versión revisada del acta."
            "Si el usuario no proporciona información para un campo específico, debes responder con el valor 'información no proporcionada'."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI(
    model="gpt-3.5-turbo-16k", max_tokens=2000
)

class Attendee(TypedDict):
    name: Annotated[str, ..., "Nombre del asistente"]
    position: Annotated[str, ..., "Cargo del asistente"]
    role: Annotated[str, ..., "Función del asistente en la reunión"]

class Task(TypedDict):
    responsible: Annotated[str, ..., "Persona responsable de la tarea"]
    date: Annotated[str, ..., "Fecha límite de la tarea"]
    description: Annotated[str, ..., "Descripción específica de la tarea"]

class MeetingMinutes(TypedDict):
    title: Annotated[str, ..., "Título de la reunión"]
    date: Annotated[str, ..., "Fecha de la reunión"]
    attendees: Annotated[List[Attendee], ..., "Lista de asistentes a la reunión"]
    summary: Annotated[str, ..., "Resumen de las actas en 3 párrafos separados por saltos de línea"]
    takeaways: Annotated[List[str], ..., "Puntos clave de las actas de la reunión"]
    conclusions: Annotated[List[str], ..., "Conclusiones y acciones a tomar"]
    next_meeting: Annotated[List[str], ..., "Compromisos adquiridos para la próxima reunión"]
    tasks: Annotated[List[Task], ..., "Lista de tareas específicas con responsables y fechas"]
    message: Annotated[str, ..., "Mensaje de respuesta a los comentarios del crítico"]

llm_with_tools = llm.bind_tools(tools)

structured_llm = meeting_minutes_prompt | llm_with_tools.with_structured_output(MeetingMinutes)

async def generate_meeting_minutes(messages: List[SystemMessage | HumanMessage | AIMessage], transcript_path):
    try:
        meeting_minutes = await structured_llm.ainvoke({
            "messages": messages,
            "TranscriptReader": {"file_path": transcript_path}
        })
        return AIMessage(content=str(meeting_minutes))
    except Exception as e:
        error_message = f"An error occurred while generating the meeting minutes: {str(e)}"
        return AIMessage(content=error_message)
