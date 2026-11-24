
from langchain_core.tools import tool
from ddgs import DDGS
from retriever import retrieve
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


@tool
def rag_tool(query:str)->str:
    """Cerca nei documenti i passaggi per rispondere alla query. Usa il tool quando la domanda riguarda il contenuto dei documenti indicizzati"""
    try:
        docs = retrieve(query)
    except Exception as e:
        return f"Errore nel recupero dei documenti {e}"
    if not docs:
        return "Nessun informazione al riguardo trovata nei documenti"
    return "\n\n".join(
        f"[{d.metadata.get('source','?')}, pagina {int(d.metadata.get('pages',0))}]\n{d.page_content}" for d in docs)