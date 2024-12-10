import os
from langgraph_sdk import get_client
from typing import Optional, Tuple
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import json

load_dotenv()

async def create_new_thread(client) -> Optional[str]:
    """Create a new thread for each minutes processing."""
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
        
        assistant_id = os.getenv("ASSISTANT_ID_MINUTES")
        if not assistant_id:
            print("Error: ASSISTANT_ID_MINUTES environment variable not found")
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
                # Obtener el último mensaje de tipo 'ai'
                ai_messages = [msg for msg in messages if msg.get('type') == 'ai']
                if ai_messages:
                    last_ai_message = ai_messages[-1]
                    content = last_ai_message.get('content', '')
                    try:
                        # Intentar parsear como JSON para mejor formato
                        acta = json.loads(content)
                        print("\n📄 Última versión del acta:")
                        print(json.dumps(acta, ensure_ascii=False, indent=2))
                    except json.JSONDecodeError:
                        # Si no es JSON, mostrar como texto plano
                        print("\n📄 Última respuesta:")
                        print(content)
    except Exception as e:
        print(f"\n❌ Error processing event: {str(e)}")

async def process_minutes(client, assistant_id: str, minutes_text: str):
    """Process minutes text with a new thread and handle streaming."""
    thread_id = await create_new_thread(client)
    if not thread_id:
        print("❌ Error: Could not create new thread")
        return

    print("\n🔍 Procesando acta...")
    initial_state = {
        "messages": [
            HumanMessage(content=minutes_text)
        ]
    }
    
    try:
        print("⏳ Generando respuesta...")
        stream = client.runs.stream(
            assistant_id=assistant_id,
            thread_id=thread_id,
            input=initial_state,
            stream_mode="values",
            interrupt_before=["human_critique"]
        )
        
        async for event in stream:
            await process_stream_event(event)
            
    except Exception as e:
        print(f"\n❌ Error durante el streaming: {str(e)}")
    finally:
        if 'stream' in locals():
            await stream.aclose()
        print("\n✅ Procesamiento completado")

async def main():
    print("\n🚀 Iniciando sistema de actas...")
    assistant_id, client = await initialize_assistant()
    if not all([assistant_id, client]):
        print("❌ Error: Failed to initialize. Exiting.")
        return

    print("\n👋 Bienvenido al sistema de actas de reunión")
    print("ℹ️  Puede escribir 'salir' en cualquier momento para terminar")
    
    while True:
        print("\n📝 Por favor, ingrese el acta de la reunión:")
        minutes_text = input().strip()
        
        if minutes_text.lower() == 'salir':
            print("\n👋 Gracias por usar el sistema de actas.")
            break
            
        if minutes_text:
            await process_minutes(client, assistant_id, minutes_text)
        else:
            print("⚠️  Por favor, ingrese un texto válido.")

if __name__ == "__main__":
    asyncio.run(main())