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
        assistant_id = os.getenv("ASSISTANT_ID_ENGINEER")
        if not assistant_id:
            print("Error: ASSISTANT_ID_ENGINEER environment variable not found")
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
        print(f"Thread created successfully: {thread}")
        
        company_name = input("Enter company name: ").strip()
        
        initial_state = {
            "report": "",
            "documents": {},
            "messages": [{
                "role": "user",
                "content": f"Generate a app for create a new product"
            }],
            "research_count": 0
        }
        
        #print(f"\nGenerating report for {company_name}...\n")
        
        try:
            # Crear el mensaje inicial del usuario
            await client.messages.create(
                thread_id=thread["thread_id"],
                role="user",
                content=f"I want to create a new product for {company_name}. Here are the main steps I envision:\n"
                        "1. Market Research\n"
                        "2. Product Requirements\n"
                        "3. Design Phase\n"
                        "4. Development\n"
                        "5. Testing\n"
                        "6. Launch Strategy\n"
                        "Please help me create a detailed plan for this product."
            )

            # Iniciar el run
            run = await client.runs.create(
                thread_id=thread["thread_id"],
                assistant_id=assistant_id
            )

            # Streaming de la respuesta
            stream = client.runs.stream(
                assistant_id=assistant_id,
                thread_id=thread["thread_id"],
                run_id=run["run_id"],
                stream_mode="updates",
            )
            
            async for chunk in stream:
                if chunk.event == "updates" and chunk.data:
                    for key in chunk.data:
                        if 'messages' in chunk.data[key]:
                            for message in chunk.data[key]['messages']:
                                if 'content' in message:
                                    print("\n" + message['content'])
                                    print("\n" + "="*80 + "\n")

        except Exception as e:
            print(f"Error durante la ejecución: {str(e)}")
            import traceback
            print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())