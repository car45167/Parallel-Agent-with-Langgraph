
from schemas import AgentState,AgentInput
from langgraph.types import Send
from langgraph.graph import END
from config import llm
from langchain.messages import HumanMessage,SystemMessage




def route(state: AgentState):
    """Fan-out sui task pianificati.

    - Nessun task  → stringa "aggregator" (il grafo prosegue verso il join).
    - N task       → lista di N Send, uno per agente: LangGraph esegue
                     le branch in parallelo e poi le fonde nel parent.

    Ogni Send inietta SOLO {"task": <testo del task>} nel branch,
    quindi l'agente target riceve lo stato con il proprio task isolato.
    """
    tasks = state.get("tasks", [])
    parallel_tasks = [t for t in tasks if t.agent != "file_saver_agent"]
    if not parallel_tasks:
        print("Nessun task parallelo: vado dritto all'aggregator")
        return "aggregator"
    return [Send(t.agent, {"current_task": t.task}) for t in parallel_tasks]


def route_after_aggregator(state: AgentState):
    """Se il planner ha previsto un salvataggio è necessario utilizzare questo route_aggregator."""
    save_task = [t for t in state.get("tasks", []) if t.agent == "file_saver_agent"]
    if not save_task:
        return END

    original_task = save_task[0].task
    content = f"{original_task}. Contenuto da salvare {state['final_response']}"
    return Send("file_saver_agent", {"current_task":content})



def create_coding_agent(state: AgentInput):
    print("[Coding Agent]")
    system_prompt = f"Sei un agente specializzato nel coding per rispondere alle eventuali richieste dell'utente. Scrivi in modo breve il codice."
    messages = [SystemMessage(content = system_prompt), HumanMessage(content= state["current_task"])]
    response = llm.invoke(messages)
    return {"results":[{"agent":"coding_agent","content":response.content}]}


def aggregator(state: AgentState) -> AgentState:
    """Aggrega i risultati della ricerca e li restituisce all'utente."""
    agents = {r["agent"]: r["content"]  for r in state["results"]}
    print("[Aggregator]")
    parts = []
    for name in sorted(agents):
        parts.append(f"=== {name.upper()} ===\n{agents[name]}") ##lookup del valore associato a ciascuna chiave name ovvero il contenuto fornito da ciascun agente.

    return {"final_response": "\n\n".join(parts)}
