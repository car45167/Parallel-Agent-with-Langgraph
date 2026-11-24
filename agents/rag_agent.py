
from tools import rag_tool
from agents.factory_agent import factoryAgent
from config import MAX_TOOL_CALLS


tool_list =[rag_tool]
RAG_PROMPT= ("Sei un agente RAG con il solo compito di utilizzare il tool rag_tool a disposizione per il recupero di informazioni richieste dall'utente.\n" 
"Nel caso in cui le informazioni ricercate non siano presenti nei documenti, dillo in modo esplicito. Sii breve e conciso nel riassumere le informazioni\n")

rag_agent = factoryAgent("RAG Agent",RAG_PROMPT,tool_list, MAX_TOOL_CALLS)