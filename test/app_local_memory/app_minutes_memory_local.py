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

def read_meeting_file(file_path: str) -> str:
    """Lee el contenido de un archivo de transcripción de reunión."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        raise Exception(f"Error al leer el archivo: {str(e)}")

async def process_minutes(minutes_text: str):
    """Process minutes text and handle user interactions."""
    print("\n🔍 Procesando acta...")
    
    initial_state = {
        "messages": [HumanMessage(content=minutes_text)],
    }
    
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    revision_count = 0
    
    try:
        # Análisis preliminar y ciclo de revisión
        print("⏳ Analizando puntos clave y acciones...")
        result = await graph.ainvoke(initial_state, config, interrupt_before=["revise_keypoints", "generate"])
        
        while True:
            content = result["messages"][-1].content
            try:
                analysis = json.loads(content)
                print("\n📊 Análisis de puntos clave:")
                print(json.dumps(analysis, ensure_ascii=False, indent=2))
            except json.JSONDecodeError:
                print("\n📊 Análisis de puntos clave:")
                print(content)

            print("\n📝 ¿Desea modificar los puntos clave? (Ingrese sus cambios o 'aprobado' para continuar):")
            user_feedback = input().strip()
            
            if user_feedback.lower() == "aprobado" or user_feedback == "":
                # Continuamos con la generación del acta
                result = await graph.ainvoke(result, config, interrupt_before=["human_critique"])
                break
                
            # Actualizar con el feedback del usuario
            print("\n🔄 Actualizando análisis...")
            new_state = {
                "messages": result["messages"] + [HumanMessage(content=user_feedback)]
            }
            # Interrumpimos antes de revise_keypoints y generate para mantener el ciclo de revisión
            result = await graph.ainvoke(new_state, config, interrupt_before=["revise_keypoints", "generate"])

        # Obtenemos el contenido del acta generada
        content = result["messages"][-1].content
        try:
            acta = json.loads(content)
            print("\n📄 Borrador inicial del acta:")
            print(json.dumps(acta, ensure_ascii=False, indent=2))
        except json.JSONDecodeError:
            print("\n📄 Borrador inicial:")
            print(content)

        # Ciclo de revisión del acta
        while True:
            print("\n📝 Por favor, ingrese sus comentarios sobre el acta (o 'aprobado' si está conforme):")
            user_comments = input().strip()
            
            if user_comments.lower() == "aprobado" or user_comments == "":
                print("\n✨ Generando versión final aprobada...")
                try:
                    acta = json.loads(content)
                    print("\n📋 ACTA FINAL APROBADA:")
                    print(json.dumps(acta, ensure_ascii=False, indent=2))
                except json.JSONDecodeError:
                    print("\n📋 VERSIÓN FINAL APROBADA:")
                    print(content)
                print("\n✅ Proceso de acta completado")
                break
            
            # Actualizar con los comentarios del usuario
            revision_count += 1
            print(f"\n🔄 Procesando revisión #{revision_count}...")
            new_state = {
                "messages": result["messages"] + [HumanMessage(content=user_comments)]
            }
            
            result = await graph.ainvoke(new_state, config, interrupt_before=["human_critique"])
            content = result["messages"][-1].content
            
            try:
                acta = json.loads(content)
                print(f"\n📝 Revisión #{revision_count} del acta:")
                print(json.dumps(acta, ensure_ascii=False, indent=2))
            except json.JSONDecodeError:
                print(f"\n📝 Revisión #{revision_count}:")
                print(content)

    except Exception as e:
        print(f"\n❌ Error durante el procesamiento: {str(e)}")
    
    print("\n✅ Procesamiento completado")

async def main():
    print("\n🚀 Iniciando sistema de actas...")
    print("\n👋 Bienvenido al sistema de actas de reunión")
    print("ℹ️  Puede escribir 'salir' en cualquier momento para terminar")
    
    while True:
        print("\n📝 Por favor, seleccione una opción:")
        print("1. Ingresar texto directamente")
        print("2. Cargar archivo .txt")
        print("3. Salir")
        
        option = input("\nOpción: ").strip()
        
        if option == "3" or option.lower() == "salir":
            print("\n👋 Gracias por usar el sistema de actas.")
            break
            
        minutes_text = ""
        
        if option == "1":
            print("\n📝 Por favor, ingrese el acta de la reunión:")
            minutes_text = input().strip()
        elif option == "2":
            print("\n📂 Por favor, ingrese la ruta del archivo .txt:")
            file_path = input().strip()
            try:
                minutes_text = read_meeting_file(file_path)
                print(f"\n📄 Archivo cargado exitosamente: {file_path}")
            except Exception as e:
                print(f"\n❌ Error al cargar el archivo: {str(e)}")
                continue
        else:
            print("\n⚠️ Opción no válida. Por favor, intente nuevamente.")
            continue
            
        if minutes_text:
            await process_minutes(minutes_text)
        else:
            print("⚠️ Por favor, ingrese un texto válido.")

if __name__ == "__main__":
    asyncio.run(main())