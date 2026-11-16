
from schemas import AgentPrivateState
from config import llm, MAX_TOOL_CALLS
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode, tools_condition
from schemas import AgentInput, AgentOutput




### Factory per la definizione di agenti con tool-> La richiamo ogni volta che ho un agente che deve richiamare qualche tool
def factoryAgent(display_name: str, system_prompt: str, tool_list: list, max_tool_call: int=MAX_TOOL_CALLS):
    llm_bind_tools = llm.bind_tools(tool_list)

    def agent_node(state: AgentPrivateState):
        print(f"[{display_name}]")
        message =[SystemMessage(content=system_prompt),HumanMessage(content= state["current_task"])]+ state.get("messages",[])

        ## Nel caso in cui si superano i tentativi previsti chiama llm per la risposta finale.
        if state.get("tools_call_amount",0)>= max_tool_call:
            print(f"[{display_name}] limite di tool call raggiunto, risposta senza tool")
            message.append(HumanMessage(content="Hai raggiunto il limite di ricerche, rispondi subito alla richiesta con le informazioni raccolte."))

            result = llm.invoke(message)
            return {"messages":[result]}

        response = llm_bind_tools.invoke(message)
        return {"messages": [response], "tools_call_amount": 1 if response.tool_calls else 0} ## tools_call_amount è la chiave di AgentPrivateState mentre tool_calls è l'attibuto indentificativo dei tools fissato da langchain

    def publish(state: AgentPrivateState):
        ##Scrive in result l'ultima risposta
        last_message = state["messages"][-1].content
        return {"results":[{"agent": display_name, "content":last_message or "Nessuna risposta dal modello"}] }


    subGraph = StateGraph(AgentPrivateState, input_schema = AgentInput, output_schema = AgentOutput)
    subGraph.add_node("agent", agent_node)
    subGraph.add_node("tools",ToolNode(tool_list))
    subGraph.add_node("publish",publish)
    subGraph.add_edge(START,"agent")
    subGraph.add_conditional_edges("agent",tools_condition,{"tools":"tools", END: "publish"})
    subGraph.add_edge("tools","agent")
    subGraph.add_edge("publish", END)
    return subGraph.compile()


