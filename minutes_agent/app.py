import requests
import json
import os
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Obtener la URL base y el ID del hilo desde las variables de entorno
#BASE_URL = os.getenv('BASE_URL', 'http://localhost:8123')
#THREAD_ID = os.getenv('THREAD_ID', 'default_thread_id')

def invoke_graph(initial_state):
    url = f"http://localhost:8123/threads/123e4567-e89b-12d3-a456-426614174000/runs"

    payload = {
        "input": [initial_state],
        "metadata": {},
        "config": {
            "tags": [None],
            "configurable": {}
        },
        "interrupt_before": "*",
        "interrupt_after": "*",
        "stream_mode": ["values"],
        "on_disconnect": "cancel",
        "feedback_keys": [None],
        "multitask_strategy": "reject"
    }
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Lanza una excepción para códigos de estado HTTP no exitosos
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error al hacer la solicitud: {e}")
        return None

def main():
    initial_state = {
        "audioFile": "",
        "transcript": "Crea una acta de reunión ficticia sobre la creación de un nuevo producto de software para una empresa de tecnología. No tengo la transcripción.",
        "wordCount": 100,
        "critique": "",
        "approved": False,
        "minutes": "",
        "outputFormatMeeting": "",
        "messages": [],
    }

    result = invoke_graph(initial_state)
    
    if result:
        print("Respuesta del servidor:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
        # Aquí puedes agregar lógica adicional para procesar el resultado
        # Por ejemplo, extraer el acta de la reunión si está disponible
        if 'minutes' in result:
            print("\nActa de la reunión:")
            print(result['minutes'])
    else:
        print("No se pudo obtener una respuesta del servidor.")

if __name__ == "__main__":
    main()