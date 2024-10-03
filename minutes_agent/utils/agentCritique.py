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
        return """
              
        You are an expert reviewer of meeting minutes who writes fluently in Spanish. Your task is to review the minutes provided and give concise feedback on how to improve them. Follow the instructions carefully:

        1. Read the minutes provided by the user.
        
        2. Read the transcript of the meeting provided by the user to check that everything relevant to the meeting has been included.
        
        3. Evaluate the minutes using the following criteria
           - Clarity and conciseness
           - Structure and organisation
           - Inclusion of relevant information
           - Accuracy in recording decisions and actions
           - Grammar and spelling
        
        4. If you find areas that need improvement, be brief and specific about what needs to be corrected. Be direct and precise in your suggestions. Include your feedback in <feedback> tags.
        
        5. If you think the minutes are good and need no correction, reply with the word "None" without any additional text.
        
        Remember that your sole purpose is to provide useful and concise feedback to improve the minutes. Do not add unnecessary praise or explanations.
        
        Give your response in Spanish.
        
        """

    def get_content(self, minutes: Dict[str, Any], transcript: str) -> str:
        today = date.today().strftime("%d/%m/%Y")
        return f"""
        Please review the following meeting minutes with the corresponding meeting transcript.
 
            Minutes of the meeting:
            -----
            {json.dumps(minutes, indent=2)}
            -----
    
            Meeting transcript:
            -----
            Transcript:
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
