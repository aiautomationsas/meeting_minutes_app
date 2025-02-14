import asyncio
import os
import sys
from dotenv import load_dotenv

# Añadir el directorio raíz al PYTHONPATH
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

# Adjust the import path to match your project structure
from complience_officer.main import graph

async def main():
    # Load environment variables
    load_dotenv()

    # Example queries about SARLAFT or SAGRILAFT
    queries = [
        "futbol",
        "¿Cuáles son los principales requisitos del SAGRILAFT para empresas en Colombia?",
        "Explique las obligaciones de una empresa bajo el sistema SARLAFT",
        "¿Cómo implementar un sistema de gestión de riesgos de lavado de activos?"
    ]

    # Iterate through queries and run the workflow
    for query in queries:
        print(f"\n--- Procesando consulta: {query} ---")
        
        # Initial state
        initial_state = {
            "user_query": query,
            "documents": {},
            "messages": [{"role": "human", "content": query}]
        }

        # Run the workflow
        try:
            result = await graph.ainvoke(initial_state)
            
            # Print the generated report
            if 'messages' in result:
                for msg in result['messages']:
                    if hasattr(msg, 'content'):
                        print(msg.content)
            
            # Print the final report if available
            if 'report' in result:
                print("\n--- Informe Final ---")
                print(result['report'])
        
        except Exception as e:
            print(f"Error procesando la consulta: {e}")
        
        # Optional: Add a small delay between queries
        await asyncio.sleep(2)

if __name__ == "__main__":
    asyncio.run(main()) 