from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
from langgraph.types import Command
import json

from company_research_interrupt.state.types import ResearchState
from company_research_interrupt.nodes.tools import tools, tools_by_name

# Initialize model
model = ChatOpenAI(model="gpt-4-turbo-preview", temperature=0).bind_tools(tools)

async def tool_node(state: ResearchState):
    """Process tool calls and store results."""
    docs = state.get('documents', {})
    tool_messages = []
    
    last_message = state["messages"][-1]
    if not hasattr(last_message, "tool_calls"):
        return {
            "messages": [],
            "documents": docs,
            "awaiting_review": False,
            "research_complete": False
        }
    
    for tool_call in last_message.tool_calls:
        try:
            tool = tools_by_name[tool_call["name"]]
            results = await tool.ainvoke(tool_call["args"]["query"])
            
            new_docs = json.loads(results)
            docs_str = ""
            
            for doc in new_docs:
                if doc['url'] not in docs:
                    docs[doc['url']] = doc
                    title = doc.get('title', 'Sin título')
                    content = doc.get('content', doc.get('snippet', 'Sin contenido'))
                    docs_str += f"\nFuente: {doc['url']}\nTítulo: {title}\nContenido: {content}\n"
            
            tool_messages.append(
                ToolMessage(
                    content=docs_str or "No se encontraron resultados relevantes.",
                    tool_call_id=tool_call["id"]
                )
            )
        except Exception as e:
            print(f"Error processing tool call: {str(e)}")
            tool_messages.append(
                ToolMessage(
                    content="Error al procesar la búsqueda.",
                    tool_call_id=tool_call["id"]
                )
            )
    
    return {
        "messages": tool_messages,
        "documents": docs,
        "research_count": state.get('research_count', 0) + 1,
        "awaiting_review": True,
        "research_complete": False
    }

def human_review_sources(state: ResearchState) -> Command[Literal["generate_report"]]:
    """Allow human to review and filter sources before continuing."""
    docs = state.get('documents', {})
    
    print("\n👀 Revisión de fuentes encontradas:")
    for url, doc in docs.items():
        print(f"\n🔗 {url}")
        print(f"📑 {doc.get('title', 'Sin título')}")
    
    while True:
        print("\n¿Desea continuar con estas fuentes? (s/n):")
        response = input().strip().lower()
        
        if response == 's':
            return Command(
                goto="generate_report",
                update={
                    "awaiting_review": False,
                    "research_complete": True,
                    "messages": [SystemMessage(content="Generate report using only the approved sources.")],
                    "documents": docs
                }
            )
        elif response == 'n':
            print("\nSeleccione las URLs a mantener (separadas por coma):")
            selected_urls = [url.strip() for url in input().strip().split(',')]
            filtered_docs = {
                url: doc 
                for url, doc in docs.items() 
                if url in selected_urls
            }
            
            return Command(
                goto="generate_report",
                update={
                    "documents": filtered_docs,
                    "awaiting_review": False,
                    "research_complete": True,
                    "messages": [SystemMessage(content="Generate report using only the approved sources.")]
                }
            )
        else:
            print("Por favor, responda 's' o 'n'")

def call_model(state: ResearchState):
    """Generate research queries."""
    prompt = """You are an expert researcher tasked with preparing a detailed company report.
    Use the tavily_search tool to find relevant information.
    Make ONE search at a time and wait for the results before making additional searches.
    Focus on recent news, financial data, and significant developments."""
    
    messages = state.get('messages', []) + [SystemMessage(content=prompt)]
    response = model.invoke(messages)
    return {"messages": [response]}

def write_report(state: ResearchState):
    """Generate final report."""
    docs = state.get('documents', {})
    
    print("\n📚 Generando informe con las siguientes fuentes:")
    for url, doc in docs.items():
        print(f"\n🔗 {url}")
        print(f"📑 {doc.get('title', 'Sin título')}")
    
    sources_text = ""
    urls_text = "\nFuentes consultadas:\n"
    for url, doc in docs.items():
        sources_text += f"\nURL: {url}\n"
        sources_text += f"Título: {doc.get('title', 'Sin título')}\n"
        sources_text += f"Contenido: {doc.get('content', doc.get('snippet', 'Sin contenido'))}\n"
        sources_text += "-" * 50 + "\n"
        urls_text += f"- {url}\n"
    
    prompt = f"""Por favor, genera un informe detallado de la empresa utilizando EXCLUSIVAMENTE la información de las siguientes fuentes aprobadas:

{sources_text}

IMPORTANTE:
1. Usa ÚNICAMENTE la información de las fuentes proporcionadas arriba
2. NO agregues información externa o de otras fuentes
3. Si algún dato no está en las fuentes proporcionadas, NO lo incluyas
4. Enfócate en:
   - Datos financieros recientes
   - Posición en el mercado
   - Desarrollos significativos
5. Estructura el informe de manera clara y profesional
6. Responde en español
7. INCLUYE AL FINAL DEL INFORME la siguiente sección de fuentes:

{urls_text}"""
    
    messages = [SystemMessage(content=prompt)]
    response = model.invoke(messages)
    
    return {
        "messages": [AIMessage(content=response.content)],
        "report": response.content,
        "research_complete": True,
        "awaiting_review": False
    }