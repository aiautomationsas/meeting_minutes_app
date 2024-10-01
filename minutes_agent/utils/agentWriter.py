from typing import Dict, Any
from langchain_cohere import ChatCohere
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
import json
from datetime import date
from dotenv import load_dotenv
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

from minutes_agent.utils.state import MinutesGraphState, MinutesContent, Attendee, Task

class WriterAgent:
    def __init__(self):
        self.llm = ChatCohere(
            model="command-r-plus",
            temperature=0
        )
        self.parser = JsonOutputParser(pydantic_object=MinutesContent)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _invoke_chain(self, chain, messages):
        return await chain.ainvoke({"messages": messages})

    async def writer(self, state: MinutesGraphState) -> MinutesContent:
        prompt = self.create_prompt('write')
        return await self.generate_minutes(state, prompt)

    async def revise(self, state: MinutesGraphState) -> MinutesContent:
        prompt = self.create_prompt('revise')
        return await self.generate_minutes(state, prompt)

    def create_prompt(self, type: str) -> ChatPromptTemplate:
        format_instructions = self.parser.get_format_instructions()
        system_message = self.get_system_message(type, format_instructions)
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            MessagesPlaceholder(variable_name="messages")
        ])

    def get_system_message(self, type: str, format_instructions: str) -> str:
        base_message = f"""Como experto en {'creación de actas de reuniones' if type == 'write' else 'revisión de actas de reuniones'}, eres un chatbot diseñado para {'facilitar el proceso de generación de actas de reuniones de manera eficiente' if type == 'write' else 'mejorar las actas basándote en la crítica proporcionada'}.

        {format_instructions}

        Responde en español.
        No inventes información. {'Si no encuentras información en la transcripción, simplemente responde "Información no proporcionada" para cada caso.' if type == 'write' else ''}
        Asegúrate de que tus respuestas estén estructuradas, sean concisas y proporcionen una visión general completa de los procedimientos de la reunión para un registro efectivo y acciones de seguimiento.
        Responde solo con el objeto JSON, sin texto adicional."""

        return base_message

    async def generate_minutes(self, state: MinutesGraphState, prompt: ChatPromptTemplate) -> MinutesContent:
        chain_writer = prompt | self.llm | self.parser
        content = self.create_content(state)
        request_message = HumanMessage(content=content)

        try:
            await asyncio.sleep(1)
            result = await self._invoke_chain(chain_writer, [request_message])
            print("Resultado parseado:", result)
            return result
        except Exception as error:
            print('Error en generate_minutes:', error)
            print('Detalles del error:', str(error))
            raise ValueError(f'No se pudieron generar las actas: {str(error)}')

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

    async def run(self, state: MinutesGraphState) -> MinutesContent:
        if state.get("critique") and state["critique"].strip() != "":
            return await self.revise(state)
        else:
            return await self.writer(state)