from typing import Dict, Any
from langchain_cohere import ChatCohere  # Asegúrate de que la importación es correcta
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage, AIMessage
import json
from datetime import date
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential

from .state import MinutesGraphState

from dotenv import load_dotenv

load_dotenv()

class CritiqueAgent:
    def __init__(self):
        self.llm = ChatCohere(
            model="command-r-plus",
            temperature=0
        )
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", self.get_system_prompt()),
            MessagesPlaceholder(variable_name="messages")
        ])

    def get_system_prompt(self) -> str:
        return """Eres crítico de las actas de reuniones. Tu único propósito es proporcionar comentarios breves sobre las actas de la reunión para que el escritor sepa qué corregir.

        Responde en español.
        Si crees que las actas de la reunión son buenas, por favor devuelve solo la palabra 'None' sin ningún texto adicional."""

    def get_content(self, minutes: Dict[str, Any], transcript: str) -> str:
        today = date.today().strftime("%d/%m/%Y")
        return f"""La fecha de hoy es {today}. Estas son las actas de la reunión:
        -----
        {json.dumps(minutes, indent=2)}
        -----

        Tu tarea es proporcionar comentarios sobre las actas de la reunión solo si es necesario.
        Asegúrate de que se den nombres para las votaciones divididas y para el debate.
        Se debe nombrar al proponente de cada moción.
        
        Esta es la transcripción de la reunión:
        -----
        Transcripción:
        {transcript}
        -----
        """

    def process_critique_result(self, result: AIMessage) -> str:
        if hasattr(result, 'content'):
            return result.content.strip()
        return "None"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _invoke_chain(self, chain_critique, messages):
        return await chain_critique.ainvoke({"messages": messages})

    async def critique(self, state: MinutesGraphState) -> Dict[str, Any]:
        chain_critique = self.prompt | self.llm
        content = self.get_content(state['minutes'], state['transcript'])
        request_message = HumanMessage(content=content)

        try:
            # Añadir un pequeño retraso antes de la llamada a la API
            await asyncio.sleep(1)
            result = await self._invoke_chain(chain_critique, [request_message])
            print("Respuesta cruda del modelo:", result)
            critique = self.process_critique_result(result)
            return {**state, "critique": critique}
        except Exception as error:
            print('Error en critique:', error)
            print('Detalles del error:', str(error))
            raise ValueError('No se pudo generar la crítica')
