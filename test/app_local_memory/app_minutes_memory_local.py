import sys
import os
from typing import List
import asyncio
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

# Añadir el directorio raíz al PYTHONPATH
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from meeting_minutes_agent.minutes_agent_local_memory import graph

load_dotenv()

async def main():
    initial_state = {
        "messages": [HumanMessage(content="Crea el acta de una reunión ficticia sobre planeación estratégica")],
    }
    
    config = {"configurable": {"thread_id": "9"}}
    resultado = await graph.ainvoke(initial_state, config)
    print("Resultado final:", resultado)

if __name__ == "__main__":
    asyncio.run(main())