from typing_extensions import Annotated, TypedDict

class TranscriptReader(TypedDict):
    """Lee el contenido de un archivo de transcript."""
    file_path: Annotated[str, ..., "Ruta del archivo del transcript"]

def read_transcript(file_path: str) -> str:
    """Lee el contenido de un archivo de transcript."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Error al leer el archivo: {str(e)}"

tools = [TranscriptReader]