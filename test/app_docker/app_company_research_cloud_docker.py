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
    """Process streaming events, focusing on the last AI response."""
    try:
        if hasattr(event, 'data') and isinstance(event.data, dict):
            messages = event.data.get('messages', [])
            if messages:
                # Obtener el último mensaje
                last_message = messages[-1]
                if last_message.get('type') == 'ai':
                    content = last_message.get('content', '')
                    if content:
                        try:
                            # Intentar parsear como JSON para mejor formato
                            reporte = json.loads(content)
                            print("\n📄 Reporte generado:")
                            print(json.dumps(reporte, ensure_ascii=False, indent=2))
                        except json.JSONDecodeError:
                            # Si no es JSON, mostrar como texto plano
                            print("\n📄 Reporte generado:")
                            print(content)
    except Exception as e:
        print(f"\n❌ Error processing event: {str(e)}")

async def process_company_research(client, assistant_id: str, company_name: str):
    """Process company research with a new thread."""
    thread_id = await create_new_thread(client)
    if not thread_id:
        print("❌ Error: Could not create new thread")
        return

    print("\n🔍 Investigando empresa...")
    initial_state = {
        "report": "",
        "documents": {},
        "messages": [HumanMessage(content=f"Generate a detailed report about {company_name}. Respond in Spanish.")],
        "research_count": 0
    }
    
    try:
        # Procesamiento inicial
        print("⏳ Iniciando investigación...")
        stream = client.runs.stream(
            assistant_id=assistant_id,
            thread_id=thread_id,
            input=initial_state,
            stream_mode="values"
        )
        
        async for event in stream:
            await process_stream_event(event)
            
    except Exception as e:
        print(f"\n❌ Error durante el proceso: {str(e)}")
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