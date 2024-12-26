import sys
import os
from typing import List
import asyncio
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import json
import uuid

# Add root directory to PYTHONPATH
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from company_research_interrupt.company_research_interrupt import graph

load_dotenv()

async def process_company_research(company_name: str):
    """Process company research with user review of sources."""
    print("\n🔍 Investigando empresa...")
    
    initial_state = {
        "report": "",
        "documents": {},
        "messages": [HumanMessage(content=f"Generate a detailed report about {company_name}")],
        "research_count": 0,
        "awaiting_review": False,
        "research_complete": False
    }
    
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    try:
        print("⏳ Iniciando investigación...")
        
        # Primera ejecución hasta el interrupt
        result = await graph.ainvoke(initial_state, config, interrupt_before=["human_review"])
        
        while True:
            # Obtener el siguiente estado
            snapshot = graph.get_state(config)
            
            if not snapshot or not hasattr(snapshot, "next"):
                break
                
            # Continuar la ejecución
            result = await graph.ainvoke(None, config, interrupt_before=["human_review"])
            
            if result.get("report"):
                print("\n📊 Reporte Final:")
                print(result["report"])
                break
                
    except Exception as e:
        print(f"\n❌ Error durante el procesamiento: {str(e)}")
    
    print("\n✅ Investigación completada")

async def main():
    print("\n🚀 Iniciando sistema de investigación empresarial...")
    print("\n👋 Bienvenido al sistema de investigación de empresas")
    print("ℹ️  Puede escribir 'salir' en cualquier momento para terminar")
    
    while True:
        print("\n🏢 Por favor, ingrese el nombre de la empresa a investigar:")
        company_name = input().strip()
        
        if company_name.lower() == 'salir':
            print("\n👋 Gracias por usar el sistema de investigación empresarial.")
            break
            
        if company_name:
            await process_company_research(company_name)
        else:
            print("⚠️  Por favor, ingrese un nombre de empresa válido.")

if __name__ == "__main__":
    asyncio.run(main())
