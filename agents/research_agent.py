from tools import research_tool
from agents.factory_agent import factoryAgent

RESEARCH_PROMPT = (
        "Sei un agente di ricerca web. Per rispondere alle domande dell'utente "
        "usa il tool 'research_tool'. Non attingere alla tua conoscenza interna: "
        "usa solo le informazioni ottenute dal tool. Se non trovi nulla rispondi "
        "'Non ho trovato informazioni al riguardo'."
    )


research_tools = [research_tool]

research_agent = factoryAgent("Research Agent", RESEARCH_PROMPT,research_tools)
