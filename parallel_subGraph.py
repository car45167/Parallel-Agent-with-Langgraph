import os
from typing import Annotated, Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
import operator
from ddgs import DDGS
from langchain_core.tools import tool
from langgraph.types import Send
from langgraph.prebuilt import ToolNode,tools_condition
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
import uuid
import logging


load_dotenv()
ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CF_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")

MAX_TOOL_CALLS = 2

llm = ChatOpenAI(
    model="@cf/qwen/qwen3.8-27b",
    base_url=f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/v1",
    api_key=CF_TOKEN,
    max_tokens=4096,
    max_retries=0,
)
logger = logging.getLogger(__name__)   

### Classi per decidere le task da assegnare a ciascun agent.
class PlannedTask(BaseModel):
    agent: Literal["research_agent","coding_agent"] = Field(description="Agente a cui assegnare un task")
    task: str = Field(
        description="Lista dei task da assegnare agli agenti. Vuota se non serve nessun agente."
    )

class AgentResult(TypedDict):
    agent: str
    content: str

class AgentState(TypedDict):
    tasks: list[PlannedTask] ##Task da assegnare a ciascun agente
    task : str ##Input utente
    results: Annotated[list[AgentResult], operator.add]   ##Risultato di ogni agente- risorsa condivisa 
    final_response: str  ## risposta finale dell'aggregator


class AgentPrivateState(TypedDict):
    results: Annotated[list[AgentResult], operator.add]
    current_task: str
    messages: Annotated[list[BaseMessage],add_messages] ## Serve per il richiamo dei tool di ciascun agente in quanto per convenzione usano messages
    tool_calls: Annotated[int, operator.add] ## Lo uso per contare quante volte chiamano i tools e porre un limite. 



### Factory per la definizione di agenti con tool-> La richiamo ogni volta che ho un agente che deve richiamare qualche tool
def factoryAgent(name: str, system_prompt: str, tool_list: list, max_tool_call: int=MAX_TOOL_CALLS):
    llm_bind_tools = llm.bind_tools(tool_list)

    def agent_node(state: AgentPrivateState):
        print(f"[agent]:{name}")
        message =[SystemMessage(content=system_prompt),HumanMessage(content= state["current_task"])]+ state.get("messages",[])

        ## Nel caso in cui si superano i tentativi previsti chiama llm per la risposta finale.
        if state.get("tool_calls",0)>= max_tool_call:
            print(f"{name} limite di tool call raggiunto, risposta senza tool")
            message.append(HumanMessage(content="Hai raggiunto il limite di ricerche, rispondi subito alla richiesta con le informazioni raccolte."))

            result = llm.invoke(message)
            return {"messages":[result]}

        response = llm_bind_tools.invoke(message)
        return {"messages": [response], "tool_calls": 1 if response.tool_calls else 0}

    def publish(state: AgentPrivateState):
        ##Scrive in result l'ultima risposta
        last_message = state["messages"][-1].content
        return {"results":[{"agent": name, "content":last_message or "Nessuna risposta dal modello"}] }


    subGraph = StateGraph(AgentPrivateState)
    subGraph.add_node("agent", agent_node)
    subGraph.add_node("tools",ToolNode(tool_list))
    subGraph.add_node("publish",publish)
    subGraph.add_edge(START,"agent")
    subGraph.add_conditional_edges("agent",tools_condition,{"tools":"tools", END: "publish"})
    subGraph.add_edge("tools","agent")
    subGraph.add_edge("publish", END)
    return subGraph.compile()



@tool
def research_tool(query: str) -> str:
    """Questo tool è utilizzato per eseguire ricerche sul web """
    try:
        ddgs = DDGS()
        results = ddgs.text(query, max_results=3)
        
        # Se non ci sono risultati, restituiamo una stringa invece di lasciare results vuoto
        if not results:
            return "Nessun risultato trovato sul web per questa query."
            
        # Estraiamo in modo sicuro il 'body' (evitando KeyError se la chiave manca)
        return "\n\n".join(r.get("body", "") for r in results if "body" in r)
    except Exception as e:
        # Catturiamo qualsiasi errore (es. rate limit, timeout, ecc.)
        return f"Errore durante la ricerca web: {str(e)}"




RESEARCH_PROMPT = (
        "Sei un agente di ricerca web. Per rispondere alle domande dell'utente "
        "usa il tool 'research_tool'. Non attingere alla tua conoscenza interna: "
        "usa solo le informazioni ottenute dal tool. Se non trovi nulla rispondi "
        "'Non ho trovato informazioni al riguardo'."
    )

research_tools = [research_tool]

research_agent = factoryAgent("researcher", RESEARCH_PROMPT,research_tools)

###


def create_coding_agent(state: AgentState) -> AgentState:
    print("Sono nel coding agent")
    system_prompt = f"Sei un agente specializzato nel coding per rispondere alle eventuali richieste dell'utente. Scrivi in modo breve il codice."
    messages = [SystemMessage(content = system_prompt), HumanMessage(content= state["current_task"])]
    response = llm.invoke(messages)
    return {"results":[{"agent":"coding_agent","content":response.content}]}



def route(state: AgentState):
    """Fan-out sui task pianificati.

    - Nessun task  → stringa "aggregator" (il grafo prosegue verso il join).
    - N task       → lista di N Send, uno per agente: LangGraph esegue
                     le branch in parallelo e poi le fonde nel parent.

    Ogni Send inietta SOLO {"task": <testo del task>} nel branch,
    quindi l'agente target riceve lo stato con il proprio task isolato.
    """
    tasks = state.get("tasks", [])
    if not tasks:
        print("Nessun task: vado dritto all'aggregator")
        return "aggregator"
    return [Send(t.agent, {"current_task": t.task}) for t in tasks]


#######################
# Main Graph
#######################
class Plan(BaseModel):
    tasks: list[PlannedTask] = Field(default_factory=list)


planner_llm = llm.with_structured_output(Plan)

PLANNER_PROMPT = (
    "Sei un planner. Dividi la richiesta dell'utente in compiti, uno per agente, "
    "senza sovrapposizioni.\n"
    "- research: SOLO domande su fatti, persone, eventi, notizie.\n"
    "- coding: SOLO scrittura o correzione di codice.\n"
    "Usa un agente solo se la richiesta contiene un compito per lui. "
    "Lista vuota SOLO per saluti o chiacchiere.\n"
    "Esempio: 'chi era Dijkstra, scrivi l'algoritmo' -> "
    "research: 'Chi era Edsger Dijkstra?'; "
    "coding: 'Scrivi l'algoritmo di Dijkstra per i cammini minimi.'"
)


def plannerAgent(state:AgentState) -> AgentState:
    """Questo agente pianifica le azioni da eseguire in base all'input dell'utente. """

    print("Sono nel planner...")
    # 2. L'input dell'utente va nel HumanMessage
    human_msg = HumanMessage(content=state['task'])
    
    plan = planner_llm.invoke([SystemMessage(PLANNER_PROMPT), human_msg])
    tasks = plan.tasks if plan else []##Visto che è BaseModel posso accedere tramite il '.' . Se fosse stato TypeDict dovevo accedere con state['valore']
    print(f"Task pianificati: {[(t.agent, t.task) for t in tasks]}")
    return{"tasks": tasks}



def aggregator(state: AgentState) -> AgentState:
    """Aggrega i risultati della ricerca e li restituisce all'utente."""
    agents = {r["agent"]: r["content"]  for r in state["results"]}
    print("Sono nell'aggregatore...")
    parts = []
    for name in sorted(agents):
        parts.append(f"=== {name.upper()} ===\n{agents[name]}")

    return {"final_response": "\n\n".join(parts)}


mainGraph = StateGraph(AgentState)

mainGraph.add_node("planner_agent", plannerAgent)
mainGraph.add_node("aggregator", aggregator)
mainGraph.add_node("research_agent", research_agent)
mainGraph.add_node("coding_agent", create_coding_agent)
mainGraph.add_edge(START, "planner_agent")
mainGraph.add_conditional_edges(
    "planner_agent",
    route,
    ["research_agent", "coding_agent", "aggregator"],  # target validi
)

mainGraph.add_edge("coding_agent",  "aggregator")
mainGraph.add_edge("aggregator", END)

checkpointer = InMemorySaver()

mainGraph = mainGraph.compile(checkpointer=checkpointer)




if __name__ == "__main__":
        user_input = input(" Inserisci la tua richiesta:")
        
        config = { 'configurable': {'thread_id': str(uuid.uuid4())} }
        
        result = mainGraph.invoke({"task": user_input}, config=config)
        print(f"Risposta finale: {result['final_response']}")

   