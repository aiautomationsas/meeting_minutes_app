import sys
import os
from typing import List
import asyncio
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import json
import uuid

# Añadir el directorio raíz al PYTHONPATH
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root_dir)

from company_research_agent.company_researcher import app as graph

load_dotenv()

async def process_company_research(company_name: str):
    """Process company research and handle user interactions."""
    print("\n🔍 Investigando empresa...")
    
    initial_state = {
        "report": "",
        "documents": {},
        "messages": [HumanMessage(content=f"Generate a detailed report about {company_name}. Respond in Spanish.")],
        "research_count": 0
    }
    
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    try:
        # Procesamiento inicial
        print("⏳ Generando investigación inicial...")
        resultado = await graph.ainvoke(initial_state, config)
        
        # Mostrar resultado formateado
        try:
            # Intentar parsear como JSON para mejor formato
            reporte = json.loads(resultado["messages"][-1].content)
            print("\n📄 Reporte generado:")
            print(json.dumps(reporte, ensure_ascii=False, indent=2))
        except json.JSONDecodeError:
            # Si no es JSON, mostrar como texto plano
            print("\n📄 Reporte generado:")
            print(resultado["messages"][-1].content)

        # Ciclo de feedback del usuario
        while True:
            print("\n📝 Por favor, ingrese sus comentarios o solicitudes adicionales (o 'Aprobado' si está conforme):")
            user_comments = input().strip()
            
            if user_comments.lower() == "aprobado" or user_comments == "":
                print("\n✅ Reporte aprobado y proceso completado")
                break
            
            # Actualizar el estado con los comentarios del usuario
            print("\n🔄 Procesando comentarios...")
            new_state = {
                "report": resultado.get("report", ""),
                "documents": resultado.get("documents", {}),
                "messages": resultado["messages"] + [HumanMessage(content=user_comments)],
                "research_count": 0  # Reiniciar contador para nueva búsqueda
            }
            
            resultado = await graph.ainvoke(new_state, config)
            
            # Mostrar resultado actualizado
            try:
                reporte = json.loads(resultado["messages"][-1].content)
                print("\n📄 Reporte actualizado:")
                print(json.dumps(reporte, ensure_ascii=False, indent=2))
            except json.JSONDecodeError:
                print("\n📄 Reporte actualizado:")
                print(resultado["messages"][-1].content)

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