from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, Literal
import operator
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage




### Classi per decidere le task da assegnare a ciascun agent.
class PlannedTask(BaseModel):
    agent: Literal["research_agent","coding_agent","file_saver_agent"] = Field(description="Agente a cui assegnare un task")
    task: str = Field(
        description="Singolo compito che l'agente deve svolgere."
    )

class AgentResult(TypedDict):
    agent: str
    content: str

class AgentState(TypedDict):
    tasks: list[PlannedTask] ##Task da assegnare a ciascun agente
    user_input : str ##Input utente
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