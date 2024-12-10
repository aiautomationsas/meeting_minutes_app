import os
from langgraph_sdk import get_client
from typing import Optional, Tuple
import asyncio
from dotenv import load_dotenv

load_dotenv()

async def initialize_assistant() -> Tuple[Optional[str], Optional[dict], Optional[object]]:
    try:
        client = get_client(url="http://localhost:8123/")
        
        # Obtener el assistant_id del env
        assistant_id = os.getenv("ASSISTANT_ID_MINUTES")
        if not assistant_id:
            print("Error: ASSISTANT_ID_MINUTES environment variable not found")
            return None, None, None
        
        # Create thread
        thread = await client.threads.create()
        
        return assistant_id, thread, client
    except Exception as e:
        print(f"Error initializing assistant: {str(e)}")
        return None, None, None

async def main():
    assistant_id, thread, client = await initialize_assistant()
    if assistant_id and thread and client:
        print("Bienvenido al sistema de actas de reunión")
        print("\nPor favor, ingrese el acta de la reunión:")
        minutes_text = input().strip()
        
        initial_state = {
            "messages": [{
                "role": "user",
                "content": minutes_text
            }]
        }
        
        print("\nProcesando el acta...\n")
        
        while True:
            # Create a new stream for each iteration
            stream = None
            try:
                print("\n=== Iniciando Stream ===")
                stream = client.runs.stream(
                    assistant_id=assistant_id,
                    thread_id=thread["thread_id"],
                    input=initial_state,
                    stream_mode="values",
                    interrupt_before=["human_critique"],
                )
                
                final_answer = None
                print("\nProcesando stream...")
                async for chunk in stream:
                    print(f"\nEvento recibido: {chunk.event}")
                    if chunk.event == "values":
                        final_answer = chunk.data
                        print(f"Datos recibidos: {final_answer}")
                
                # Get user feedback when the stream is interrupted
                try:
                    user_feedback = input("\nPor favor, ingrese su retroalimentación: ").strip()
                    
                    # Si el feedback está vacío, cambiarlo a "Aprobado"
                    if not user_feedback:
                        user_feedback = "Aprobado"
                    
                    print(f"\n=== Actualizando estado con feedback: {user_feedback} ===")
                    # Update the state with the user feedback using the correct format
                    await client.threads.update_state(
                        thread_id=thread["thread_id"],
                        values={"messages": {"role": "user", "content": user_feedback}},
                        as_node="human_critique"
                    )
                    
                    print("\n=== Iniciando Stream de Continuación ===")
                    # Continue the stream execution after the update
                    continuation_stream = client.runs.stream(
                        assistant_id=assistant_id,
                        thread_id=thread["thread_id"],
                        input=None,  # No input needed for continuation
                        stream_mode="values",
                        interrupt_before=["human_critique"]
                    )
                    
                    final_answer = None
                    print("\nProcesando stream de continuación...")
                    async for chunk in continuation_stream:
                        print(f"\nEvento recibido: {chunk.event}")
                        if chunk.event == "values":
                            final_answer = chunk.data
                            print(f"Datos recibidos: {final_answer}")
                    
                    # Si la respuesta es "Aprobado", cerrar el programa después de procesar
                    if user_feedback == "Aprobado":
                        print("\nFinalizando el programa...")
                        break
                        
                except KeyboardInterrupt:
                    print("\nFinalizando el programa...")
                    break
            finally:
                if stream:
                    await stream.aclose()

if __name__ == "__main__":
    asyncio.run(main())