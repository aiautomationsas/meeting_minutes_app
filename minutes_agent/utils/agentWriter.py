from typing import List
from langchain_cohere import ChatCohere 
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
import json
from datetime import date
from dotenv import load_dotenv
from langchain.output_parsers import JsonOutputParser
from langchain.pydantic_v1 import BaseModel, Field

load_dotenv()

from minutes_agent.utils.state import MinutesGraphState, MinutesContent

class WriterAgent:
    def __init__(self):
        self.llm = ChatCohere(
            model="command-r-plus",
            temperature=0
        )
        self.output_parser = JsonOutputParser(pydantic_object=MinutesSchema)

    async def writer(self, state: MinutesGraphState) -> MinutesContent:
        prompt = self.create_prompt('write')
        return await self.generate_minutes(state, prompt)

    async def revise(self, state: MinutesGraphState) -> MinutesContent:
        prompt = self.create_prompt('revise')
        return await self.generate_minutes(state, prompt)

    def create_prompt(self, type: str) -> ChatPromptTemplate:
        format_instructions = self.get_format_instructions()
        system_message = self.get_system_message(type, format_instructions)
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages")
        ])

    def get_format_instructions(self) -> str:
        return self.output_parser.get_format_instructions()

    def get_system_message(self, type: str, format_instructions: str) -> str:
        base_message = f"""Como experto en {'creación de actas de reuniones' if type == 'write' else 'revisión de actas de reuniones'}, eres un chatbot diseñado para {'facilitar el proceso de generación de actas de reuniones de manera eficiente' if type == 'write' else 'mejorar las actas basándote en la crítica proporcionada'}.

        {format_instructions}

        Explicación de los campos:
        -----
        "title": Título de la reunión,
        "date": Fecha de la reunión,
        "attendees": Array de objetos que representan a los asistentes a la reunión: Cada objeto debe contener las claves: "name", "position" y "role". La clave "role" indica la función del asistente en la reunión. Si alguno de estos valores no está claro o no se menciona, se debe asignar el valor por defecto "none".,
        "summary": "Resume sucintamente las actas de la reunión en 3 párrafos claros y coherentes. Separa los párrafos usando caracteres de nueva línea.",
        "takeaways": Lista de los puntos clave de las actas de la reunión,
        "conclusions": Lista de conclusiones y acciones a tomar,
        "next_meeting": Lista de los compromisos adquiridos en la reunión. Asegúrate de revisar todo el contenido de la reunión antes de dar tu respuesta,
        "tasks": Lista de diccionarios para los compromisos adquiridos en la reunión. Los diccionarios deben tener los siguientes valores clave "responsible", "date" y "description". En el valor clave "description", es aconsejable mencionar específicamente qué se espera que haga la persona responsable en lugar de indicar acciones generales. Asegúrate de incluir todos los elementos de la lista next_meeting,
        "message": Mensaje para enviar al crítico en respuesta a cada uno de sus comentarios,
        -----
        Responde en español.
        No inventes información. {'Si no encuentras información en la transcripción, simplemente responde "Información no proporcionada" para cada caso.' if type == 'write' else ''}
        Asegúrate de que tus respuestas estén estructuradas, sean concisas y proporcionen una visión general completa de los procedimientos de la reunión para un registro efectivo y acciones de seguimiento.
        Responde solo con el objeto JSON, sin texto adicional."""

        return base_message

    async def generate_minutes(self, state: MinutesGraphState, prompt: ChatPromptTemplate) -> MinutesContent:
        chain_writer = prompt | self.llm | self.output_parser
        content = self.create_content(state)
        request_message = HumanMessage(content=content)

        try:
            result = await chain_writer.ainvoke({"messages": [request_message]})
            return result
        except Exception as error:
            print('Error en generate_minutes:', error)
            raise ValueError('No se pudieron generar las actas')

    def create_content(self, state: MinutesGraphState) -> str:
        today = date.today().strftime("%d/%m/%Y")
        if state.get("critique") and state.get("minutes"):
            return f"""La fecha de hoy es {today}. Esta es una crítica de una reunión.
                    -----
                    {state['critique']}.
                    -----
                    Tu tarea será escribir las actas corregidas de la reunión, 
                    teniendo en cuenta cada uno de los comentarios de la crítica y las actas a corregir. 
                    Debe estar dividido en párrafos usando caracteres de nueva línea.
                    También tendrás acceso a la transcripción de la reunión.
            #####
            actas a corregir:
            {json.dumps(state['minutes'])}
            #####
            #####
            crítica:
            {state['critique']}
            #####
            #####
            transcripción:
            {state['transcript']}
            #####"""
        else:
            return f"""La fecha de hoy es {today}. Esta es una transcripción de una reunión.
                    -----
                    {state['transcript']}.
                    -----
                    Tu tarea es redactar para mí las actas de la reunión descrita arriba,
                    incluyendo todos los puntos de la reunión.
                    Las actas de la reunión deben tener aproximadamente {state['wordCount']} palabras
                    y deben estar divididas en párrafos usando caracteres de nueva línea."""

    def extract_json(self, string: str) -> str:
        try:
            return json.loads(string)
        except json.JSONDecodeError:
            # Si falla, intentamos extraer manualmente
            json_start = string.find('{')
            json_end = string.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                return string[json_start:json_end + 1]
            raise ValueError("No se pudo extraer un JSON válido de la respuesta")

    async def run(self, state: MinutesGraphState) -> MinutesContent:
        if state.get("critique") and state["critique"].strip() != "":
            return await self.revise(state)
        else:
            return await self.writer(state)

# Definición del esquema Pydantic para la estructura de las actas
class Attendee(BaseModel):
    name: str
    position: str
    role: str

class Task(BaseModel):
    responsible: str
    date: str
    description: str

class MinutesSchema(BaseModel):
    title: str = Field(description="Título de la reunión")
    date: str = Field(description="Fecha de la reunión")
    attendees: List[Attendee] = Field(description="Lista de asistentes a la reunión")
    summary: str = Field(description="Resumen de las actas de la reunión en 3 párrafos claros y coherentes")
    takeaways: List[str] = Field(description="Puntos clave de las actas de la reunión")
    conclusions: List[str] = Field(description="Conclusiones y acciones a tomar")
    next_meeting: List[str] = Field(description="Compromisos adquiridos en la reunión")
    tasks: List[Task] = Field(description="Lista de tareas con responsable, fecha y descripción")
    message: str = Field(description="Mensaje para enviar al crítico en respuesta a sus comentarios")