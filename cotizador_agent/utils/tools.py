def read_transcript(file_path: str) -> str:
    """Lee el contenido de un archivo de transcript."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Error al leer el archivo: {str(e)}"
