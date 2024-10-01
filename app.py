import requests

url = "http://localhost:8123/runs/stream"

payload = {
    "assistant_id": "815a2f1b-4fd7-40f3-a16a-f4482a3593e7",
    "input": {
        "audioFile": "",
        "transcript": "Crea una acta de reunión ficticia sobre la creación de un nuevo producto de software para una empresa de tecnología. No tengo la transcripción.",
        "wordCount": 100,
        "critique": "",
        "approved": False,
        "minutes": "",
        "outputFormatMeeting": "",
        "messages": []
    },
    "metadata": {},
    "config": {
        "tags": [],
        "configurable": {}
    },
    "interrupt_before": ["human_critique"],
    "stream_mode": ["values"]
}
headers = {"Content-Type": "application/json"}

response = requests.post(url, json=payload, headers=headers)

if response.status_code == 200:
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            print(decoded_line)
else:
    print(f"Error en la solicitud. Código de estado: {response.status_code}")
    print("Contenido de la respuesta:", response.text)