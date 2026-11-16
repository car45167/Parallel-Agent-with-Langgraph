from tools import save_file
from agents.factory_agent import factoryAgent
from config import PATH


FILE_SAVER_PROMPT=f"Sei un agente specializzato nel salvare informazioni su file utilizzando il tool 'save_file'. Puoi lavorare solo nella cartella corrispondente a {PATH}"
"Il nome del file di testo sei libero di inventarlo a patto che sia opportunamento correlato al task assegnato."


file_saver_tools = [save_file]


file_saver_agent = factoryAgent("File Saver Agent", FILE_SAVER_PROMPT, file_saver_tools)
