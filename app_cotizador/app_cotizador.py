"""
Esta app invoca al agente de cotización para generar una propuesta de servicios. No usa langgraph sdk
"""

import sys
from pathlib import Path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

from dotenv import load_dotenv
from pathlib import Path
from langchain_core.messages import SystemMessage, HumanMessage
from cotizador_agent import graph

load_dotenv()

def read_document(file_path):
    """Lee el contenido de un archivo markdown."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        raise Exception(f"Error al leer el archivo {file_path}: {str(e)}")

def get_documents_content():
    """Lee y combina el contenido de todos los documentos markdown en la carpeta documents."""
    current_dir = Path(__file__).parent
    documents_dir = current_dir / 'documents'
    
    content_parts = []
    print("\n=== Documentos encontrados ===")
    print(f"Buscando documentos en: {documents_dir}")
    
    if not documents_dir.exists():
        print(f"Error: El directorio {documents_dir} no existe")
        return None
    
    md_files = list(documents_dir.glob('*.md'))
    if not md_files:
        print(f"No se encontraron archivos .md en {documents_dir}")
        return None
        
    for file_path in md_files:
        try:
            section_title = file_path.stem.replace('_', ' ').upper()
            content = read_document(file_path)
            print(f"\nArchivo: {file_path.name}")
            print(f"Título: {section_title}")
            print(f"Contenido:\n{content}")
            print("-" * 50)
            content_parts.append(f"\n### {section_title}:\n{content}")
        except Exception as e:
            print(f"Error al leer {file_path.name}: {str(e)}")
            continue
    
    return "\n".join(content_parts) if content_parts else None

def print_stream(stream):
    """Imprime el stream de mensajes del agente."""
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(f"User: {message[1]}")
        else:
            print(f"Assistant: {message.content}")

def main():
    try:
        # Leer el contenido de los documentos
        documents_content = get_documents_content()
        if not documents_content:
            print("\nError: No se pudieron leer los documentos necesarios.")
            return

        # Solicitar al usuario la línea de negocio
        print("\n=== Selección de Línea de Negocio ===")
        print("Por favor, seleccione la línea de negocio para la cotización:")
        print("1. Productividad Operacional")
        print("2. Transformación Digital")
        
        while True:
            try:
                option = input("\nIngrese el número de la opción (1 o 2): ")
                if option == "1":
                    business_line = "operacional_productivity"
                    break
                elif option == "2":
                    business_line = "transformacion_digital"
                    break
                else:
                    print("Opción no válida. Por favor, seleccione 1 o 2.")
            except ValueError:
                print("Entrada no válida. Por favor, ingrese 1 o 2.")

        # Solicitar al usuario el resultado esperado
        print("\nPor favor, describe el resultado esperado de la cotización:")
        print("(Por ejemplo: 'Necesito una propuesta que incluya costos por hora, tiempo estimado y entregables')")
        print("(Presiona Ctrl+D para finalizar la entrada)")
        
        expected_result = []
        try:
            while True:
                line = input()
                expected_result.append(line)
        except EOFError:
            expected_result = "\n".join(expected_result)

        # Crear el mensaje inicial con el contenido de los documentos y el resultado esperado
        initial_message = f"""Please help me generate a quote based on the following information:
        ================================================
        BUSINESS LINE:
        {business_line}

        COMPANY INFORMATION AND PAINS:
        {documents_content}

        EXPECTED_RESULT:
        {expected_result}
        ================================================

        Be very careful with the indicators you select that must be from the {business_line} line that solve the company's problems and expected result.

        Include strategies, timelines that should not exceed 3.5 months, resources and KPIs to implement the indicators.
        """

        # Crear el estado inicial
        initial_state = {
            "messages": [
                SystemMessage(content=
                """
                You are an expert in creating quotes for companies who analyses very well the available indicators with the analyze_indicators tool.
                You aim to create company quotes with the generate_business_proposal tool.
                You ensure that all requirements are met and that the selected indicators are consistent with the company's improvement objectives.
                For the solutions of the digital transformation line do not use the word sofware but digital solutions.
                You respond in Spanish.
                """),
                HumanMessage(content=initial_message)
            ]
        }

        # Ejecutar el graph y mostrar resultados
        print("\n=== Iniciando análisis ===\n")
        print_stream(graph.stream(initial_state, stream_mode="values"))

    except Exception as e:
        print(f"Error durante la ejecución: {str(e)}")
        import traceback
        print("\nTraceback completo del error:")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()