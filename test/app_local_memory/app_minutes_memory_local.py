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

from meeting_minutes_agent.minutes_agent_cloud import graph

load_dotenv()

async def process_minutes(minutes_text: str):
    """Process minutes text and handle user interactions."""
    print("\n🔍 Procesando acta...")
    
    initial_state = {
        "messages": [HumanMessage(content=minutes_text)],
    }
    
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    try:
        # Procesamiento inicial
        print("⏳ Generando respuesta inicial...")
        resultado = await graph.ainvoke(initial_state, config, interrupt_before=["human_critique"])
        
        # Mostrar resultado formateado
        try:
            # Intentar parsear como JSON para mejor formato
            acta = json.loads(resultado["messages"][-1].content)
            print("\n📄 Acta generada:")
            print(json.dumps(acta, ensure_ascii=False, indent=2))
        except json.JSONDecodeError:
            # Si no es JSON, mostrar como texto plano
            print("\n📄 Acta generada:")
            print(resultado["messages"][-1].content)

        # Ciclo de críticas del usuario
        while True:
            print("\n📝 Por favor, ingrese sus comentarios (o 'Aprobado' si está conforme):")
            user_comments = input().strip()
            
            if user_comments.lower() == "aprobado" or user_comments == "":
                print("\n✅ Acta aprobada y proceso completado")
                break
            
            # Actualizar el estado con los comentarios del usuario
            print("\n🔄 Procesando comentarios...")
            new_state = {
                "messages": resultado["messages"] + [HumanMessage(content=user_comments)]
            }
            
            resultado = await graph.ainvoke(new_state, config, interrupt_before=["human_critique"])
            
            # Mostrar resultado actualizado
            try:
                acta = json.loads(resultado["messages"][-1].content)
                print("\n📄 Acta actualizada:")
                print(json.dumps(acta, ensure_ascii=False, indent=2))
            except json.JSONDecodeError:
                print("\n📄 Acta actualizada:")
                print(resultado["messages"][-1].content)

    except Exception as e:
        print(f"\n❌ Error durante el procesamiento: {str(e)}")
    
    print("\n✅ Procesamiento completado")

async def main():
    print("\n🚀 Iniciando sistema de actas...")
    print("\n👋 Bienvenido al sistema de actas de reunión")
    print("ℹ️  Puede escribir 'salir' en cualquier momento para terminar")
    
    while True:
        print("\n📝 Por favor, ingrese el acta de la reunión:")
        minutes_text = input().strip()
        
        if minutes_text.lower() == 'salir':
            print("\n👋 Gracias por usar el sistema de actas.")
            break
            
        if minutes_text:
            await process_minutes(minutes_text)
        else:
            print("⚠️  Por favor, ingrese un texto válido.")

if __name__ == "__main__":
    asyncio.run(main())