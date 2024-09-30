import json
from langgraph.graph import StateGraph, END, START
#from langgraph.checkpoint.memory import MemorySaver
from minutes_agent.utils.nodes import read_transcript, create_minutes, create_critique, human_critique, human_approved, output_meeting
from minutes_agent.utils.state import MinutesGraphState



def should_continue(state: MinutesGraphState) -> str:
    critique = state.get('critique', '')
    if critique == "":
        return "human_approved"
    return "create_minutes"

# Crear el grafo
workflow = (
    StateGraph(MinutesGraphState)
    .add_node("read_transcript", read_transcript)
    .add_node("create_minutes", create_minutes)
    .add_node("create_critique", create_critique)
    .add_node("human_critique", human_critique)
    .add_node("human_approved", human_approved)
    .add_node("output_meeting", output_meeting)
    .add_edge(START, "read_transcript")
    .add_edge("read_transcript", "create_minutes")
    .add_edge("create_minutes", "create_critique")
    .add_edge("create_critique", "human_critique")
    .add_conditional_edges("human_critique", should_continue)
    .add_edge("human_approved", "output_meeting")
    .add_edge("output_meeting", END)
)

#checkpointer = MemorySaver()
app =  workflow.compile()


#app = workflow.compile(checkpointer=checkpointer, 
#                       interrupt_before=["human_critique"])


#def run_graph():
#    initial_state = {
#        "audioFile": "",
#        "transcript": "Crea una acta de reunión ficticia sobre la creación de un nuevo producto de software para una empresa de tecnología. No tengo la transcripción",
#        "wordCount": 100,
#        "critique": "",
#        "approved": False,
#        "minutes": "",
#        "outputFormatMeeting": "",
#        "messages": [],
#    }

#    try:
#        final_state = app.invoke(
#            initial_state,
#            config={"configurable": {"thread_id": 42}}  # Puedes ajustar el thread_id según sea necesario
#        )

        # Imprimir el resultado final
#        print("Estado final:")
#        print(json.dumps(final_state, indent=2))

        # Si quieres acceder a un campo específico, por ejemplo, las actas finales:
#        if "minutes" in final_state:
#           print("\nActas finales:")
#            print(final_state["minutes"])

        # Si quieres acceder al último mensaje (si existe):
#        if final_state["messages"]:
#            print("\nÚltimo mensaje:")
#            print(final_state["messages"][-1].content)

#    except Exception as e:
#        print(f"Error al ejecutar el grafo: {e}")

# Ejecutar el grafo si este script se ejecuta directamente
#if __name__ == "__main__":
#    run_graph()