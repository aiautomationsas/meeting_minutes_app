from typing import TypedDict, List, Annotated, Dict, Union, Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AnyMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, add_messages, END
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from tavily import AsyncTavilyClient
from datetime import datetime
import json
import asyncio
from dotenv import load_dotenv
import os

load_dotenv()

# Estado del agente
class RailwayState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    documents: Dict[str, Dict]
    research_count: int
    report: str

# Estructura para búsquedas en Tavily
class TavilyQuery(BaseModel):
    query: str = Field(description="consulta específica sobre normativa ferroviaria")
    topic: str = Field(description="tipo de búsqueda, usar 'general' para normativas")
    days: int = Field(description="días hacia atrás para la búsqueda")
    raw_content: bool = Field(description="incluir contenido completo de las fuentes")

class TavilySearchInput(BaseModel):
    sub_queries: List[TavilyQuery] = Field(
        description="conjunto de sub-consultas sobre normativas ferroviarias"
    )

# Configurar cliente Tavily y modelo
tavily_client = AsyncTavilyClient()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@tool("tavily_search", args_schema=TavilySearchInput)
async def tavily_search(sub_queries: List[TavilyQuery]):
    """Realiza búsquedas sobre normativas ferroviarias usando Tavily."""
    async def perform_search(query):
        try:
            response = await tavily_client.search(
                query=f"{query.query} railway standards regulations",
                topic=query.topic,
                days=query.days,
                include_raw_content=query.raw_content,
                max_results=5
            )
            return response['results']
        except Exception as e:
            print(f"Error en búsqueda Tavily: {str(e)}")
            return []

    search_tasks = [perform_search(q) for q in sub_queries]
    search_responses = await asyncio.gather(*search_tasks)
    
    all_results = []
    for response in search_responses:
        all_results.extend(response)
    
    return all_results

tools = [tavily_search]
tools_by_name = {tool.name: tool for tool in tools}
model = model.bind_tools(tools)

# Nodo para procesar herramientas
async def process_tools(state: RailwayState):
    """Procesa las búsquedas de Tavily y almacena los resultados."""
    docs = state.get('documents', {})
    docs_str = ""
    msgs = []
    
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        new_docs = await tool.ainvoke(tool_call["args"])
        
        for doc in new_docs:
            if doc['url'] not in docs:
                docs[doc['url']] = doc
                # Construir resumen usando título y snippet si están disponibles
                title = doc.get('title', 'Sin título')
                content = doc.get('content', doc.get('snippet', 'Sin contenido disponible'))
                docs_str += f"\nFuente: {doc['url']}\nTítulo: {title}\nContenido: {content}\n"
                
        msgs.append(ToolMessage(
            content=f"Documentos encontrados sobre normativas ferroviarias:\n{docs_str}",
            tool_call_id=tool_call["id"]
        ))
    
    return {
        "messages": msgs,
        "documents": docs,
        "research_count": state.get('research_count', 0)
    }

def analyze_question(state: RailwayState):
    """Analiza la pregunta y busca información relevante."""
    prompt = """Eres un experto en normativas ferroviarias internacionales.
    Analiza la pregunta del usuario y busca información específica sobre:
    1. Normas UIC, EN, ISO relevantes
    2. Regulaciones por país/región
    3. Estándares técnicos aplicables
    4. Actualizaciones recientes en normativas
    
    Usa la herramienta Tavily para encontrar información actualizada."""
    
    messages = state.get('messages', []) + [SystemMessage(content=prompt)]
    response = model.invoke(messages)
    
    return {
        "messages": [response],
        "research_count": state.get('research_count', 0)
    }

def should_continue(state: RailwayState) -> Literal["tools", "report"]:
    """Decide si continuar investigando o generar el reporte."""
    messages = state.get('messages', [])
    last_message = messages[-1]
    
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        state['research_count'] = state.get('research_count', 0) + 1
        return "tools"
    
    if state.get('research_count', 0) >= 3:
        return "report"
    
    return "report"

def generate_report(state: RailwayState):
    """Genera el reporte final basado en la información recopilada."""
    def evaluate_relevance(doc):
        """Evalúa la relevancia del documento basándose en palabras clave."""
        keywords = [
            'railway', 'ferroviario', 'UIC', 'EN', 'ISO', 'normativa',
            'regulación', 'estándar', 'seguridad', 'certificación',
            'directiva', 'reglamento', 'técnico', 'especificación'
        ]
        
        content = (doc.get('content', '') or doc.get('snippet', '')).lower()
        title = doc.get('title', '').lower()
        
        # Contar coincidencias de palabras clave
        keyword_matches = sum(1 for kw in keywords if kw.lower() in content or kw.lower() in title)
        
        # Determinar el límite de caracteres basado en la relevancia
        if keyword_matches >= 4:
            return 2000  # Documentos muy relevantes
        elif keyword_matches >= 2:
            return 1000  # Documentos moderadamente relevantes
        else:
            return 500   # Documentos menos relevantes

    # Preparar resumen de documentos
    docs_summary = []
    total_chars = 0
    max_total_chars = 50000  # Límite total para evitar exceder el contexto

    for url, doc in state.get('documents', {}).items():
        title = doc.get('title', 'Sin título')
        content = doc.get('content', doc.get('snippet', 'Sin contenido'))
        
        # Determinar límite de caracteres para este documento
        char_limit = evaluate_relevance(doc)
        
        # Ajustar el límite si nos acercamos al máximo total
        remaining_chars = max_total_chars - total_chars
        char_limit = min(char_limit, remaining_chars)
        
        if char_limit <= 0:
            break
            
        # Recortar contenido
        trimmed_content = content[:char_limit]
        if len(content) > char_limit:
            trimmed_content += "..."
            
        total_chars += len(trimmed_content)
        
        docs_summary.append({
            'url': url,
            'title': title,
            'excerpt': trimmed_content,
            'relevance_score': evaluate_relevance(doc)  # Incluir score para referencia
        })
        
        if total_chars >= max_total_chars:
            break

    prompt = f"""Basado en la información recopilada, genera un reporte detallado que incluya:
    1. Normativas aplicables
    2. Análisis de cumplimiento
    3. Recomendaciones específicas
    4. Referencias a documentos oficiales
    
    Documentos principales encontrados (ordenados por relevancia):
    {json.dumps(sorted(docs_summary, key=lambda x: x['relevance_score'], reverse=True), indent=2, ensure_ascii=False)}
    
    Responde en español y cita las fuentes específicas."""
    
    messages = [state['messages'][-1]] + [SystemMessage(content=prompt)]
    response = model.invoke(messages)
    
    return {
        "messages": [AIMessage(content=f"Reporte Generado:\n{response.content}")],
        "report": response.content,
        "research_count": state.get('research_count', 0)
    }

# Crear el grafo
workflow = StateGraph(RailwayState)

# Añadir nodos
workflow.add_node("analyze", analyze_question)
workflow.add_node("tools", process_tools)
workflow.add_node("generate_report", generate_report)

# Configurar punto de entrada
workflow.set_entry_point("analyze")

# Añadir edges condicionales
workflow.add_conditional_edges(
    "analyze",
    should_continue,
    {
        "tools": "tools",
        "report": "generate_report"
    }
)

workflow.add_edge("tools", "analyze")
workflow.add_edge("generate_report", END)

# Compilar el grafo
graph = workflow.compile()