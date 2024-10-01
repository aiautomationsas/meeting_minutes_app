from pydantic import BaseModel, Field
from typing import List, Optional
from langgraph.graph import add_messages
from langchain_core.messages import BaseMessage
from typing import Annotated, Sequence

class Attendee(BaseModel):
    name: str = Field(..., description="Nombre completo del asistente")
    position: str = Field(..., description="Cargo o posición del asistente en la organización")
    role: str = Field(..., description="Rol del asistente en la reunión (ej. moderador, presentador, participante)")

class Task(BaseModel):
    responsible: str = Field(..., description="Nombre de la persona responsable de la tarea")
    date: str = Field(..., description="Fecha límite para completar la tarea (formato: DD/MM/YYYY)")
    description: str = Field(..., description="Descripción detallada de la tarea a realizar")

class MinutesContent(BaseModel):
    title: str = Field(..., description="Título de la reunión")
    date: str = Field(..., description="Fecha en que se llevó a cabo la reunión (formato: DD/MM/YYYY)")
    attendees: List[Attendee] = Field(..., description="Lista de asistentes a la reunión")
    summary: str = Field(..., description="Resumen conciso de los principales puntos discutidos en la reunión")
    takeaways: List[str] = Field(..., description="Lista de los puntos clave o conclusiones importantes de la reunión")
    conclusions: List[str] = Field(..., description="Lista de conclusiones finales alcanzadas en la reunión")
    next_meeting: List[str] = Field(..., description="Lista de temas o puntos a tratar en la próxima reunión")
    tasks: List[Task] = Field(..., description="Lista de tareas asignadas durante la reunión")
    message: Optional[str] = Field(None, description="Mensaje opcional para el crítico o revisor de las actas")

class MinutesGraphState(BaseModel):
    audioFile: Optional[str] = Field(None, description="Ruta al archivo de audio de la reunión, si está disponible")
    transcript: str = Field(..., description="Transcripción completa de la reunión")
    wordCount: int = Field(..., description="Número aproximado de palabras esperadas en las actas")
    minutes: MinutesContent = Field(..., description="Contenido detallado de las actas de la reunión")
    critique: str = Field(..., description="Crítica o comentarios sobre las actas generadas")
    outputFormatMeeting: str = Field(..., description="Formato de salida para las actas de la reunión (ej. Markdown)")
    approved: bool = Field(..., description="Indica si las actas han sido aprobadas")
    messages: Annotated[Sequence[BaseMessage], add_messages] = Field(default_factory=list, description="Secuencia de mensajes relacionados con el proceso de generación de actas")

# Valor por defecto para messages
def default_messages() -> List[BaseMessage]:
    return []