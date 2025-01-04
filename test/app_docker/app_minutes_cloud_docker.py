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
    
    revision_count = 0
    last_content = None
    response_count = 0
    
    try:
        # Procesamiento inicial
        print("⏳ Generando borrador inicial...")
        stream = client.runs.stream(
            assistant_id=assistant_id,
            thread_id=thread_id,
            input=initial_state,
            stream_mode="values",
            interrupt_before=["human_critique"]
        )
        
        async for event in stream:
            if hasattr(event, 'data') and isinstance(event.data, dict):
                messages = event.data.get('messages', [])
                if messages:
                    ai_messages = [msg for msg in messages if msg.get('type') == 'ai']
                    if ai_messages:
                        content = ai_messages[-1].get('content', '')
                        if content != last_content:
                            last_content = content
                            response_count += 1
                            try:
                                acta = json.loads(content)
                                if response_count == 1:
                                    print("\n📄 Borrador inicial del acta:")
                                else:
                                    print("\n🤔 Borrador con reflexión AI:")
                                print(json.dumps(acta, ensure_ascii=False, indent=2))
                            except json.JSONDecodeError:
                                if response_count == 1:
                                    print("\n📄 Borrador inicial:")
                                else:
                                    print("\n🤔 Borrador con reflexión AI:")
                                print(content)
        
        # Ciclo de críticas del usuario
        while True:
            print("\n📝 Por favor, ingrese sus comentarios (o 'Aprobado' si está conforme):")
            user_comments = input().strip()
            
            # Actualizar el estado con los comentarios del usuario
            update_state = {
                "messages": [
                    HumanMessage(content=user_comments)
                ]
            }
            
            if user_comments.lower() == "aprobado" or user_comments == "":
                print("\n✨ Generando versión final aprobada...")
                # Primero actualizamos el estado
                await client.threads.update_state(
                    thread_id, 
                    update_state,
                    as_node="human_critique"
                )
                # Luego ejecutamos el stream final
                async for chunk in client.runs.stream(
                    assistant_id=assistant_id,
                    thread_id=thread_id,
                    input=None,
                    stream_mode="values"
                ):
                    if hasattr(chunk, 'data') and isinstance(chunk.data, dict):
                        messages = chunk.data.get('messages', [])
                        if messages:
                            ai_messages = [msg for msg in messages if msg.get('type') == 'ai']
                            if ai_messages:
                                content = ai_messages[-1].get('content', '')
                                if content != last_content:
                                    last_content = content
                                    try:
                                        acta = json.loads(content)
                                        print("\n📋 ACTA FINAL APROBADA:")
                                        print(json.dumps(acta, ensure_ascii=False, indent=2))
                                    except json.JSONDecodeError:
                                        print("\n📋 VERSIÓN FINAL APROBADA:")
                                        print(content)
 
                print("\n✅ Proceso de acta completado")
                break
            else:
                revision_count += 1
                print(f"\n🔄 Procesando revisión #{revision_count}...")
                await client.threads.update_state(
                    thread_id, 
                    update_state,
                    as_node="human_critique"
                )
                
                async for chunk in client.runs.stream(
                    assistant_id=assistant_id,
                    thread_id=thread_id,
                    input=None,
                    stream_mode="values",
                ):
                    if hasattr(chunk, 'data') and isinstance(chunk.data, dict):
                        messages = chunk.data.get('messages', [])
                        if messages:
                            ai_messages = [msg for msg in messages if msg.get('type') == 'ai']
                            if ai_messages:
                                content = ai_messages[-1].get('content', '')
                                if content != last_content:
                                    last_content = content
                                    try:
                                        acta = json.loads(content)
                                        print(f"\n📝 Revisión #{revision_count} del acta:")
                                        print(json.dumps(acta, ensure_ascii=False, indent=2))
                                    except json.JSONDecodeError:
                                        print(f"\n📝 Revisión #{revision_count}:")
                                        print(content)
            
    except Exception as e:
        print(f"\n❌ Error durante el streaming: {str(e)}")
        print(f"Detalles del error: {str(e)}")
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