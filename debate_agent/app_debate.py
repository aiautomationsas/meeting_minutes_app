"""
Debate Agent Application

This script runs an interactive AI debate agent where users can choose debate topics
and watch two AI debaters argue different perspectives.
"""

import sys
import traceback
from pathlib import Path
root_dir = Path(__file__).parent.parent.parent
sys.path.append(str(root_dir))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from apps.debate_agent.debate_agent import graph, debug_log
import time

load_dotenv()

def print_speaker_message(speaker: str, message: str):
    """Print a formatted message from a debate participant"""
    print(f"\n{speaker}:")
    print("=" * (len(speaker) + 1))
    print(message)
    print("-" * 80)
    # Pequeña pausa para mejor legibilidad
    time.sleep(1)

def debug_log(message: str, state: dict = None):
    """Print debug information"""
    print(f"\n🔍 DEBUG: {message}")
    if state:
        print(f"  Stage: {state.get('debate_stage')}")
        print(f"  Speaker: {state.get('current_speaker')}")
        print(f"  Topic: {state.get('topic')}")
        print(f"  Round: {state.get('debate_round', 1)}")
        print(f"  Count: {state.get('debate_count', 0)}")
        if state.get('perspectives'):
            print("  Perspectives:")
            print(f"    A: {state['perspectives'].get('a', 'Not defined')[:100]}...")
            print(f"    B: {state['perspectives'].get('b', 'Not defined')[:100]}...")
        # Mostrar el último mensaje si existe
        if state.get('messages') and len(state['messages']) > 0:
            last_message = state['messages'][-1]
            print("\n  Last Message:")
            print("  " + "-" * 40)
            print(f"  Content: {last_message.content[:200]}...")
            print("  " + "-" * 40)
    print("-" * 80)

def main():
    print("""
🎭 Bienvenido al Agente de Debates con IA 🎭

En este debate participarán:
- Moderador 🎭: Guía el debate y asegura un diálogo constructivo
- Debater A 🔵: Enfoque analítico y basado en evidencia
- Debater B 🔴: Perspectiva innovadora y desafiante

Por favor, proponga un tema para debatir.
    """)
    
    try:
        # Get the topic first
        topic = input("\n🎯 Proponga un tema para debatir: ").strip()
        if not topic:
            print("Es necesario proponer un tema para el debate.")
            return
            
        debug_log("Initializing debate with topic", {"topic": topic})
        
        # Initial state with topic
        initial_state = {
            "messages": [
                SystemMessage(content=f"El tema del debate es: {topic}")
            ],
            "topic": topic,
            "perspectives": {},
            "current_speaker": "Moderador 🎭",
            "debate_stage": "topic_selection",
            "debate_count": 0,
            "debate_round": 1
        }
        
        print(f"\n📢 Tema seleccionado: {topic}")
        print("\nComenzando el debate...")
        
        # Run the debate
        debug_log("Starting debate graph", initial_state)
        result = graph.invoke(initial_state)
        debug_log("Initial graph result", result)
        
        max_rounds = 3  # Número máximo de rondas de debate
        
        while True:
            # Print the current message
            last_ai_message = result['messages'][-1]
            current_speaker = result['current_speaker']
            print_speaker_message(current_speaker, last_ai_message.content)
            
            # Show debate information
            if result['debate_stage'] == 'debate':
                current_round = result.get('debate_round', 1)
                if current_round > max_rounds:
                    debug_log("Maximum rounds reached", result)
                    break
                    
                print(f"\n📊 Ronda de debate: {current_round}/{max_rounds}")
                debug_log(f"Debate round information", result)
            
            # Check if debate has concluded
            if result['debate_stage'] == 'conclusion':
                debug_log("Reached conclusion stage", result)
                break
                
            # Automatically continue to next stage
            result['messages'].append(HumanMessage(content="Continúe con el debate."))
            new_result = graph.invoke(result)
            debug_log("New graph result after continuation", new_result)
            result = new_result
        
        # Final conclusion
        if result['debate_stage'] == 'conclusion':
            debug_log("Presenting final conclusion", result)
            print("\n🏁 Conclusión del Debate 🏁")
            print("=" * 80)
            print_speaker_message(result['current_speaker'], result['messages'][-1].content)
        
    except Exception as e:
        debug_log(f"Error occurred: {str(e)}", result if 'result' in locals() else None)
        print("\n❌ Ha ocurrido un error:")
        print(f"Tipo de Error: {type(e).__name__}")
        print(f"Mensaje de Error: {str(e)}")
        print("\nDetalles del error:")
        traceback.print_exc()

if __name__ == "__main__":
    main() 