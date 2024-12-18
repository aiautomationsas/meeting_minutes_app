import os
from langgraph_sdk import get_client
from typing import Optional, Tuple
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import json

load_dotenv()

async def create_new_thread(client) -> Optional[str]:
    """Create a new thread for each company research."""
    try:
        thread_response = await client.threads.create()
        thread_id = thread_response["thread_id"]
        print(f"\n🔄 Nuevo thread creado: {thread_id[:8]}...")
        return thread_id
    except Exception as e:
        print(f"❌ Error creating new thread: {str(e)}")
        return None

async def initialize_assistant() -> Tuple[Optional[str], Optional[object]]:
    try:
        client = get_client(url="http://localhost:8123/")
        
        assistant_id = os.getenv("ASSISTANT_ID_RESEARCH")
        if not assistant_id:
            print("Error: ASSISTANT_ID_RESEARCH environment variable not found")
            return None, None
        
        return assistant_id, client
    except Exception as e:
        print(f"Error initializing assistant: {str(e)}")
        return None, None

async def process_stream_event(event) -> None:
    """Process streaming events from the research process."""
    try:
        if hasattr(event, 'data') and isinstance(event.data, dict):
            # Procesar documentos del estado
            if 'documents' in event.data:
                docs = event.data.get('documents', {})
                if docs:
                    # Solo mostrar las fuentes una vez, dependiendo del estado
                    if event.data.get('research_complete', False):
                        print("\n📚 Generando informe con las siguientes fuentes:")
                    elif event.data.get('awaiting_review', False):
                        print("\n👀 Revisión de fuentes encontradas:")
                    
                    for url, doc in docs.items():
                        print(f"\n🔗 {url}")
                        print(f"📑 {doc.get('title', 'Sin título')}")
                        content = doc.get('content', doc.get('snippet', 'Sin contenido'))
                        print(f"📄 {content}\n")
                        print("-" * 80)
            
            # Procesar mensajes del estado
            if 'messages' in event.data:
                messages = event.data.get('messages', [])
                if messages:
                    last_message = messages[-1]
                    
                    # Procesar mensajes del asistente
                    if last_message.get('type') == 'ai':
                        content = last_message.get('content', '')
                        if content and 'report' in event.data:
                            print("\n📊 Reporte Final:")
                            print(content)
                        elif content:
                            print("\n" + content)
            
    except Exception as e:
        print(f"\n❌ Error processing event: {str(e)}")
        print(f"Event data: {event.data if hasattr(event, 'data') else 'No data'}")

async def process_company_research(client, assistant_id: str, company_name: str):
    """Process company research with a new thread and handle streaming."""
    thread_id = await create_new_thread(client)
    if not thread_id:
        print("❌ Error: Could not create new thread")
        return

    print("\n🔍 Investigando empresa...")
    initial_state = {
        "report": "",
        "documents": {},
        "messages": [HumanMessage(content=f"Generate a detailed report about {company_name}")],
        "research_count": 0,
        "awaiting_review": False,
        "research_complete": False
    }
    
    try:
        print("⏳ Iniciando investigación...")
        current_docs = {}
        
        # Primera ejecución hasta el interrupt
        stream = client.runs.stream(
            assistant_id=assistant_id,
            thread_id=thread_id,
            input=initial_state,
            stream_mode="values",
            interrupt_before=["human_review"]
        )
        
        async for event in stream:
            await process_stream_event(event)
            
            if hasattr(event, 'data') and isinstance(event.data, dict):
                # Guardar los documentos actuales
                if 'documents' in event.data:
                    current_docs = event.data.get('documents', {})
                
                if event.data.get('awaiting_review', False):
                    while True:
                        print("\n¿Desea continuar con estas fuentes? (s/n):")
                        user_response = input().strip().lower()
                        
                        if user_response == 's':
                            filtered_docs = current_docs
                            update_state = {
                                "messages": [HumanMessage(content="Continue with all sources")],
                                "documents": filtered_docs,
                                "awaiting_review": False,
                                "research_complete": False
                            }
                            break
                        elif user_response == 'n':
                            print("\nSeleccione las URLs a mantener (separadas por coma):")
                            selected_urls = [url.strip() for url in input().strip().split(',')]
                            
                            # Filtrar documentos
                            filtered_docs = {
                                url: doc
                                for url, doc in current_docs.items()
                                if url in selected_urls
                            }
                            
                            update_state = {
                                "messages": [HumanMessage(content=f"Continue with selected sources: {', '.join(selected_urls)}")],
                                "documents": filtered_docs,  # Solo los documentos seleccionados
                                "awaiting_review": False,
                                "research_complete": False
                            }
                            break
                        else:
                            print("Por favor, responda 's' o 'n'")
                    
                    # Actualizar estado con los documentos filtrados
                    await client.threads.update_state(
                        thread_id, 
                        update_state,
                        as_node="human_review"
                    )
                    
                    # Continuar con el procesamiento
                    print("\n⏳ Generando informe final...")
                    async for chunk in client.runs.stream(
                        assistant_id=assistant_id,
                        thread_id=thread_id,
                        input=None,
                        stream_mode="values"
                    ):
                        await process_stream_event(chunk)
                    break
            
    except Exception as e:
        print(f"\n❌ Error durante el streaming: {str(e)}")
        print(f"Detalles del error: {str(e)}")
    finally:
        if 'stream' in locals():
            await stream.aclose()
        print("\n✅ Investigación completada")

async def main():
    print("\n🚀 Iniciando sistema de investigación empresarial...")
    assistant_id, client = await initialize_assistant()
    if not all([assistant_id, client]):
        print("❌ Error: Failed to initialize. Exiting.")
        return

    print("\n👋 Bienvenido al sistema de investigación de empresas")
    print("ℹ️  Puede escribir 'salir' en cualquier momento para terminar")
    
    while True:
        print("\n🏢 Por favor, ingrese el nombre de la empresa a investigar:")
        company_name = input().strip()
        
        if company_name.lower() == 'salir':
            print("\n👋 Gracias por usar el sistema de investigación empresarial.")
            break
            
        if company_name:
            await process_company_research(client, assistant_id, company_name)
        else:
            print("⚠️  Por favor, ingrese un nombre de empresa válido.")

if __name__ == "__main__":
    asyncio.run(main()) 