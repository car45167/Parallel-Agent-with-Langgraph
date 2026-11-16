
from langchain_core.tools import tool
from ddgs import DDGS

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
def save_file(path:str, content: str):
    """Questo tool è utilizzabile per salvare una risposta su un file."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
            return "Contenuto Scritto con successo!"
    except Exception as e:
        return f"Errore durante la scrittura su file: {e}"
    