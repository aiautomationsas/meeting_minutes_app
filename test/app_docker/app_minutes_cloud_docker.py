import os
from langgraph_sdk import get_client
from typing import Optional, Tuple
import asyncio
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import json

load_dotenv()

def read_text_file(file_path: str) -> Optional[str]:
    """Read content from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"❌ Error reading file: {str(e)}")
        return None

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

async def process_keypoints(client, thread_id: str, assistant_id: str) -> bool:
    last_keypoints = None
    while True:
        print("\n📋 Por favor, revise los puntos clave (escriba sus comentarios o 'aprobado' si está conforme):")
        user_input = input().strip()
        
        update_state = {
            "messages": [
                HumanMessage(content=user_input)
            ]
        }
        
        if user_input.lower() == "aprobado" or not user_input:
            await client.threads.update_state(thread_id, update_state, as_node="keypoints")
            return True
            
        await client.threads.update_state(thread_id, update_state, as_node="keypoints")
        
        async for chunk in client.runs.stream(
            assistant_id=assistant_id,
            thread_id=thread_id,
            input=None,
            stream_mode="values",
            interrupt_before=["human_keypoints"]
        ):
            if hasattr(chunk, 'data') and isinstance(chunk.data, dict):
                messages = chunk.data.get('messages', [])
                if messages:
                    ai_messages = [msg for msg in messages if msg.get('type') == 'ai']
                    if ai_messages:
                        content = ai_messages[-1].get('content', '')
                        # Solo imprimir si hay cambios
                        if content != last_keypoints:
                            print("\n🔄 Puntos clave revisados:")
                            print(content)
                            last_keypoints = content

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
    
    try:
        # Initial keypoints processing
        print("⏳ Analizando puntos clave...")
        stream = client.runs.stream(
            assistant_id=assistant_id,
            thread_id=thread_id,
            input=initial_state,
            stream_mode="values",
            interrupt_before=["human_keypoints"]
        )
        
        async for event in stream:
            if hasattr(event, 'data') and isinstance(event.data, dict):
                messages = event.data.get('messages', [])
                if messages:
                    ai_messages = [msg for msg in messages if msg.get('type') == 'ai']
                    if ai_messages:
                        content = ai_messages[-1].get('content', '')
                        print("\n📊 Puntos clave identificados:")
                        print(content)
        
        # Handle keypoints review
        if not await process_keypoints(client, thread_id, assistant_id):
            return
        
        # Continue with minutes generation and reflection
        print("\n⏳ Generando borrador del acta...")
        stream = client.runs.stream(
            assistant_id=assistant_id,
            thread_id=thread_id,
            input=None,
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
                        # Solo imprimir si hay cambios significativos
                        if content != last_content:
                            last_content = content
                            try:
                                acta = json.loads(content)
                                print("\n📄 Borrador del acta:")
                                print(json.dumps(acta, ensure_ascii=False, indent=2))
                            except json.JSONDecodeError:
                                print("\n📄 Borrador del acta:")
                                print(content)
        
        # Handle revisions
        while True:
            print("\n📝 Por favor, ingrese sus comentarios (o 'Aprobado' si está conforme):")
            user_comments = input().strip()
            
            update_state = {
                "messages": [
                    HumanMessage(content=user_comments)
                ]
            }
            
            if user_comments.lower() == "aprobado" or not user_comments:
                print("\n✨ Generando versión final aprobada...")
                await client.threads.update_state(
                    thread_id, 
                    update_state,
                    as_node="human_critique"
                )
                
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
                                try:
                                    acta = json.loads(content)
                                    print("\n📋 ACTA FINAL APROBADA:")
                                    print(json.dumps(acta, ensure_ascii=False, indent=2))
                                except json.JSONDecodeError:
                                    print("\n📋 VERSIÓN FINAL APROBADA:")
                                    print(content)
                break
            
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
                interrupt_before=["human_critique"]
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
    print("📂 Para procesar un archivo .txt, escriba 'archivo' seguido de la ruta del archivo")
    
    while True:
        print("\n📝 Por favor, ingrese el acta de la reunión o indique un archivo:")
        user_input = input().strip()
        
        if user_input.lower() == 'salir':
            print("\n👋 Gracias por usar el sistema de actas.")
            break
            
        if user_input.lower().startswith('archivo '):
            file_path = user_input[7:].strip()  # Remove 'archivo ' prefix
            if not file_path.endswith('.txt'):
                print("⚠️  Solo se aceptan archivos .txt")
                continue
                
            if not os.path.exists(file_path):
                print("⚠️  El archivo no existe")
                continue
                
            minutes_text = read_text_file(file_path)
            if minutes_text:
                print(f"📄 Procesando archivo: {file_path}")
                await process_minutes(client, assistant_id, minutes_text)
            else:
                print("⚠️  No se pudo leer el archivo")
        elif user_input:
            await process_minutes(client, assistant_id, user_input)
        else:
            print("⚠️  Por favor, ingrese un texto válido o la ruta de un archivo.")

if __name__ == "__main__":
    asyncio.run(main())
