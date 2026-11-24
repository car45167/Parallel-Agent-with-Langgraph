from schemas import AgentInput
from langchain_core.messages import HumanMessage,SystemMessage
from config import llm



def coding_agent(state: AgentInput):
    print("[Coding Agent]")
    system_prompt = f"Sei un agente specializzato nel coding per rispondere alle eventuali richieste dell'utente. Scrivi in modo breve il codice."
    messages = [SystemMessage(content = system_prompt), HumanMessage(content= state["current_task"])]
    response = llm.invoke(messages)
    return {"results":[{"agent":"coding_agent","content":response.content}]}
