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
        assistant_id = os.getenv("ASSISTANT_ID_COMPANY")
        if not assistant_id:
            print("Error: ASSISTANT_ID_COMPANY environment variable not found")
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
                "content": f"Generate a detailed report about {company_name}"
            }],
            "research_count": 0
        }
        
        print(f"\nGenerating report for {company_name}...\n")
        
        try:
            print("DEBUG: Iniciando streaming...")
            stream = client.runs.stream(
                assistant_id=assistant_id,
                thread_id=thread["thread_id"],
                input=initial_state,
                stream_mode="updates",
            )
            print("DEBUG: Stream creado")
            
            async for chunk in stream:
                if chunk.event == 'updates' and chunk.data:
                    for node, node_data in chunk.data.items():
                        if 'messages' in node_data:
                            for message in node_data['messages']:
                                if 'content' in message and message['content'].strip():
                                    print(message['content'])
                                    print("\n" + "="*80 + "\n")
        except Exception as e:
            print(f"Error durante el streaming: {str(e)}")
            print(f"Tipo de error: {type(e)}")
            import traceback
            print(traceback.format_exc())

if __name__ == "__main__":
    asyncio.run(main())