from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from typing import List
import os
from .tools import tools, read_transcript  # Importamos las herramientas

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un experto en actas de reunión evaluando un acta generada. "
            "Tienes acceso a una herramienta llamada TranscriptReader que puede leer el contenido de un archivo de transcript. "
            "Genera críticas y recomendaciones para el acta presentada, basándote en el contenido del transcript. "
            "Proporciona recomendaciones detalladas, incluyendo solicitudes de longitud, "
            "profundidad, estilo, estructura y cualquier elemento faltante o mejorable en el acta. "
            "Asegúrate de que el acta refleje fielmente el contenido del transcript."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI(model="gpt-3.5-turbo-16k", max_tokens=1000)
llm_with_tools = llm.bind_tools(tools)

reflect = reflection_prompt | llm_with_tools

async def generate_reflection(messages: List[BaseMessage], transcript_path: str) -> HumanMessage:
    try:
        cls_map = {"ai": HumanMessage, "human": AIMessage}
        # First message is the original user request. We hold it the same for all nodes
        translated = [messages[0]] + [
            cls_map[msg.type](content=msg.content) for msg in messages[1:]
        ]
        
        res = await reflect.ainvoke({
            "messages": translated,
            "TranscriptReader": {"file_path": transcript_path}
        })
        # We treat the output of this as human feedback for the generator
        return HumanMessage(content=res.content)
    except Exception as e:
        error_message = f"An error occurred while generating the reflection: {str(e)}"
        return HumanMessage(content=error_message)
