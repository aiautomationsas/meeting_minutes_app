import requests

url = "https://ht-rash-variability-38-164af1dbe40454e9bd46e5f19b24bc7c.default.us.langgraph.app/runs/stream"

payload = {
 "assistant_id": "a2c2bca2-93da-446c-bdb5-d292750bc4fb",
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
headers = {"Content-Type": "application/json",
           "X-Api-Key": "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"}

response = requests.post(url, json=payload, headers=headers)

if response.status_code == 200:
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            print(decoded_line)
else:
    print(f"Error en la solicitud. Código de estado: {response.status_code}")
    print("Contenido de la respuesta:", response.text)