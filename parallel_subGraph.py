

from langgraph.graph import StateGraph, START, END


from agents.research_agent import research_agent
from agents.rag_agent import rag_agent
from nodes.file_saver_node import file_saver_node
from agents.planner_agent import plannerAgent
from routing_nodes import aggregator, route, route_after_aggregator
from nodes.coding_node import coding_agent
from schemas import AgentState
from langgraph.checkpoint.memory import InMemorySaver
import uuid

mainGraph = StateGraph(AgentState)

mainGraph.add_node("planner_agent", plannerAgent)
mainGraph.add_node("aggregator", aggregator)
mainGraph.add_node("research_agent", research_agent)
mainGraph.add_node("file_saver_agent", file_saver_node)
mainGraph.add_node("coding_agent", coding_agent)
mainGraph.add_node("rag_agent",rag_agent)
mainGraph.add_edge(START, "planner_agent")
mainGraph.add_conditional_edges(
    "planner_agent",
    route,
    ["research_agent", "coding_agent", "aggregator", "rag_agent"],  # target validi
)
mainGraph.add_edge("rag_agent",  "aggregator")
mainGraph.add_edge("research_agent",  "aggregator")
mainGraph.add_edge("coding_agent",  "aggregator")
mainGraph.add_conditional_edges("aggregator",route_after_aggregator, ["file_saver_agent", END])
mainGraph.add_edge("file_saver_agent", END)

checkpointer = InMemorySaver()

mainGraph = mainGraph.compile(checkpointer=checkpointer)




if __name__ == "__main__":
        user_input = input(" Inserisci la tua richiesta:")
        
        config = { 'configurable': {'thread_id': str(uuid.uuid4())} }
        
        result = mainGraph.invoke({"user_input": user_input}, config=config)
        print(f"Risposta finale: {result['final_response']}")

   