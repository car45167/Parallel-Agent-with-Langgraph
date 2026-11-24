
from config import llm
from schemas import Plan, AgentState
from langchain_core.messages import HumanMessage,SystemMessage

PLANNER_PROMPT = (
    "Sei un planner. Dividi la richiesta dell'utente in compiti, uno per agente, "
    "senza sovrapposizioni.\n"
    "- research_agent: SOLO domande su fatti, persone, eventi, notizie.\n"
    "- coding_agent: SOLO scrittura o correzione di codice.\n"
    "- rag_agent: SOLO per recuperare informazioni da documenti.\n"
    "Regole sul campo 'filename':\n"
    "- Compilalo SOLO se l'utente chiede esplicitamente di salvare la risposta su un file di testo.\n"
    "- Se l'utente indica un nome per il file usa quello, altrimenti scegline uno breve e pertinente al contenuto.\n"
    "- Scrivi solo lettere, numeri e underscore, SENZA estensione: 'goku_info', non 'goku_info.txt'.\n"
    "- Se l'utente non chiede il salvataggio, lascia 'filename' a null.\n"
    "- Quando compili 'filename' devi comunque includere in 'tasks' il compito dell'agente che produce la risposta da salvare.\n"
    "Usa un agente solo se la richiesta contiene un compito per lui. Lista vuota SOLO per saluti o chiacchiere.\n"
    "Esempio 1: 'chi era Dijkstra, scrivi l'algoritmo' ->\n"
    "  tasks: research_agent: 'Chi era Edsger Dijkstra?'; coding_agent: 'Scrivi l'algoritmo di Dijkstra per i cammini minimi.'\n"
    "  filename: null\n\n"
    "Esempio 2: 'chi è Goku? salvalo su file come goku_info' ->\n"
    "  tasks: research_agent: 'Chi è Goku?'\n"
    "  filename: 'goku_info'\n\n"
    "Esempio 3: 'chi è Vegeta? salva su file di testo' ->\n"
    "  tasks: research_agent: 'Chi è Vegeta?'\n"
    "  filename: 'vegeta_info'\n"
    "Esempio 4: Recupera le informazioni sull' AI powered cancer dection dai documenti->\n"
    " tasks: rag_agent: 'Recupera le informazioni su AI powered Cancer detection'\n"
)


planner_llm = llm.with_structured_output(Plan)

def plannerAgent(state:AgentState) -> AgentState:
    """Questo agente pianifica le azioni da eseguire in base all'input dell'utente. """

    print("[Planner]")
    # 2. L'input dell'utente va nel HumanMessage
    human_msg = HumanMessage(content=state['user_input'])
    
    plan = planner_llm.invoke([SystemMessage(PLANNER_PROMPT), human_msg])
    tasks = plan.tasks if plan else []##Visto che è BaseModel posso accedere tramite il '.' . Se fosse stato TypeDict dovevo accedere con state['valore']
    filename = plan.filename
    print(f"Task pianificati: {[(t.agent, t.task) for t in tasks]}")
    print(f"Nome file: { filename}")
    return{"tasks": tasks, "filename": filename}
