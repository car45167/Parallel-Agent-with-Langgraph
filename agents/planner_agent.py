
from config import llm
from schemas import Plan, AgentState
from langchain_core.messages import HumanMessage,SystemMessage

PLANNER_PROMPT = (
    "Sei un planner. Dividi la richiesta dell'utente in compiti, uno per agente, "
    "senza sovrapposizioni.\n"
    "- research_agent: SOLO domande su fatti, persone, eventi, notizie.\n"
    "- coding_agent: SOLO scrittura o correzione di codice.\n"
    "- file_saver_agent: SOLO se viene esplicitamente richiesto il salvataggio della risposta su file di testo"
    "Usa un agente solo se la richiesta contiene un compito per lui. "
    "Lista vuota SOLO per saluti o chiacchiere.\n"
    "Esempio: 'chi era Dijkstra, scrivi l'algoritmo' -> "
    "research_agent: 'Chi era Edsger Dijkstra?'; "
    "coding_agent: 'Scrivi l'algoritmo di Dijkstra per i cammini minimi.'"
    "file_saver_agent: 'Salva la risposta finale su un file di testo.'"
)


planner_llm = llm.with_structured_output(Plan)

def plannerAgent(state:AgentState) -> AgentState:
    """Questo agente pianifica le azioni da eseguire in base all'input dell'utente. """

    print("[Planner]")
    # 2. L'input dell'utente va nel HumanMessage
    human_msg = HumanMessage(content=state['user_input'])
    
    plan = planner_llm.invoke([SystemMessage(PLANNER_PROMPT), human_msg])
    tasks = plan.tasks if plan else []##Visto che è BaseModel posso accedere tramite il '.' . Se fosse stato TypeDict dovevo accedere con state['valore']
    print(f"Task pianificati: {[(t.agent, t.task) for t in tasks]}")
    return{"tasks": tasks}
