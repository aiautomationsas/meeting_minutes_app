import json
from typing import Dict, Any, List
from minutes_agent.utils.state import MinutesGraphState, MinutesContent
from minutes_agent.utils.agentWriter import WriterAgent
from minutes_agent.utils.agentCritique import CritiqueAgent
import asyncio

writer_agent = WriterAgent()
critique_agent = CritiqueAgent()

def read_transcript(state: MinutesGraphState) -> MinutesGraphState:
    print("Leyendo transcripción...")
    print("Estado actual en read_transcript:", json.dumps(state, indent=2))
    return {**state}

def create_minutes(state: MinutesGraphState) -> MinutesGraphState:
    print("Creando o revisando acta...")
    #print("Estado actual en create_minutes:", json.dumps(state, indent=2))
    
    if not state.get('transcript'):
        raise ValueError("El transcript no puede ser null")

    minutes = asyncio.run(writer_agent.run(state))
    new_state = {**state, 'minutes': minutes}
    
    print("Nuevo estado después de create_minutes:", json.dumps(new_state, indent=2))
    return new_state

def create_critique(state: MinutesGraphState) -> MinutesGraphState:
    print("Generando crítica del acta...")
    #print("Estado actual en create_critique:", json.dumps(state, indent=2))
    
    # Llamada síncrona a la función asíncrona
    critique = asyncio.run(critique_agent.critique(state))
    new_state = {**state, 'critique': critique['critique']}
    
    print("Nuevo estado después de create_critique:", json.dumps(new_state, indent=2))
    return new_state

def human_critique(state: MinutesGraphState) -> MinutesGraphState:
    print("Revisión humana requerida...")
    print("Estado actual en human_critique:", json.dumps(state, indent=2))
    new_state = {**state}
    
    print("Nuevo estado después de human_critique:", json.dumps(new_state, indent=2))
    return new_state

def human_approved(state: MinutesGraphState) -> MinutesGraphState:
    print("Aprobando acta...")
    print("Estado actual en approve_minutes:", json.dumps(state, indent=2))
    
    new_state = {**state, 'approved': True}
    
    print("Nuevo estado después de approve_minutes:", json.dumps(new_state, indent=2))
    return new_state

def output_meeting(state: MinutesGraphState) -> MinutesGraphState:
    return {**state, 'outputFormatMeeting': generate_markdown(state['minutes'])}

def generate_markdown(minutes: MinutesContent) -> str:
    return f"""
# {minutes['title']}

**Fecha:** {minutes['date']}

## Asistentes

{generate_attendee_table(minutes['attendees']) if isinstance(minutes['attendees'], list) else minutes['attendees']}

## Resumen

{minutes['summary']}

## Puntos clave

{generate_bullet_list(minutes['takeaways']) if isinstance(minutes['takeaways'], list) else minutes['takeaways']}

## Conclusiones

{generate_bullet_list(minutes['conclusions']) if isinstance(minutes['conclusions'], list) else minutes['conclusions']}

## Próxima reunión

{generate_bullet_list(minutes['next_meeting']) if isinstance(minutes['next_meeting'], list) else minutes['next_meeting']}

## Tareas

{generate_task_table(minutes['tasks']) if isinstance(minutes['tasks'], list) else minutes['tasks']}
    """

def generate_attendee_table(attendees: List[Dict[str, str]]) -> str:
    return f"""
| **Nombre** | **Posición** | **Rol** |
|------------|--------------|---------|
{chr(10).join([f"| {attendee['name']} | {attendee['position']} | {attendee['role']} |" for attendee in attendees])}
    """

def generate_bullet_list(items: List[str]) -> str:
    return '\n'.join([f"- {item}" for item in items])

def generate_task_table(tasks: List[Dict[str, str]]) -> str:
    return f"""
| **Responsable** | **Descripción** | **Fecha** |
|-----------------|-----------------|-----------|
{chr(10).join([f"| {task['responsible']} | {task['description']} | {task['date']} |" for task in tasks])}
    """
