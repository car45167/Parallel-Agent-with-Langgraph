from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Literal,Optional
import operator
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage



## Typed Dict per gli stati del grafo-> Non fa validazione a runtime. Meno sicuro di Base Model ma è standard per gli stati interni e input del grafo anche perché i reducer operano solo su essi. BaseModel utilizzato per structured output dell llm e offre validazione e controlli rigidi a runtime( Esempio Literal vincolo che deve essere rispettato).


### Classi per decidere le task da assegnare a ciascun agent.
class PlannedTask(BaseModel):
    agent: Literal["research_agent","coding_agent","rag_agent"] = Field(description="Agente a cui assegnare un task")
    task: str = Field(
        description="Singolo compito che l'agente deve svolgere."
    )

class AgentResult(TypedDict):
    agent: str
    content: str

class AgentState(TypedDict):
    tasks: list[PlannedTask] ##Task da assegnare a ciascun agente
    user_input : str ##Input utente
    filename: Optional[str]
    results: Annotated[list[AgentResult], operator.add]   ##Risultato di ogni agente- risorsa condivisa 
    final_response: str  ## risposta finale dell'aggregator

class AgentInput(TypedDict):
    current_task: str

class AgentOutput(TypedDict):
    results: Annotated[list[AgentResult], operator.add]

class AgentPrivateState(AgentInput, AgentOutput):
    messages: Annotated[list[BaseMessage],add_messages] ## Serve per il richiamo dei tool di ciascun agente in quanto per convenzione usano messages
    tools_call_amount: Annotated[int, operator.add] ## Lo uso per contare quante volte chiamano i tools e porre un limite. 


class Plan(BaseModel):
    tasks: list[PlannedTask] = Field(description="List per gestire le task da assegnare a ciascun agente.")
    filename : Optional[str]=Field(default = None, description="Nome del file da creare su cui salvare la risposta.")
    