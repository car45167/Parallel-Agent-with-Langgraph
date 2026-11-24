
from schemas import AgentState
from langgraph.types import Send
from langgraph.graph import END


def route(state: AgentState):
    """Fan-out sui task pianificati.
    - Nessun task  → stringa "aggregator" (il grafo prosegue verso il join).
    - N task       → lista di N Send, uno per agente: LangGraph esegue
                     le branch in parallelo e poi le fonde nel parent.
    Ogni Send inietta SOLO {"task": <testo del task>} nel branch,
    quindi l'agente target riceve lo stato con il proprio task isolato.
    """
    tasks = [t for t in state.get("tasks",[])if t.task.strip()]
    if not tasks:
        print("Nessun task: vado dritto all'aggregator")
        return "aggregator"
    return [Send(t.agent, {"current_task": t.task}) for t in tasks]


def route_after_aggregator(state: AgentState):
    """Se il planner ha previsto un salvataggio è necessario utilizzare questo route_aggregator."""
    return "file_saver_agent" if state.get("filename") else END




def aggregator(state: AgentState) -> AgentState:
    """Aggrega i risultati della ricerca e li restituisce all'utente."""
    agents = {r["agent"]: r["content"]  for r in state.get("results",[])}
    print("[Aggregator]")
    parts = []
    for name in sorted(agents):
        parts.append(f"=== {name.upper()} ===\n{agents[name]}") ##lookup del valore associato a ciascuna chiave name ovvero il contenuto fornito da ciascun agente.

    return {"final_response": "\n\n".join(parts)}


