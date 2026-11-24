
import re
from pathlib import Path
from config import WORKSPACE
from schemas import AgentState

## Funzione per togliere i caratteri speciali e trasformare gli spazi in _
def _build_filename(title:str)->str:
    s = re.sub(r'[^a-zA-Z0-9]', '_', title).strip('_-')  ## tolgo _ e - dall'inizio e dalla fine della stringa.
    return f"{s or 'output'}.txt"

## Target di creazione : working directory.
def _resolve_location(filename:str)-> Path:
    location = (WORKSPACE/filename).resolve()
    try:
        location.is_relative_to(WORKSPACE.resolve()) ##Se il path scelto è un path relativo al path di base
    except ValueError:   
        raise ValueError(f"Percorso non consentito {filename!r}")
    return location


def file_saver_node (state: AgentState)->dict:
    filename = state.get("filename")
    if not filename:
        return {}

    content = state.get("final_response") or ""
    try:
        path = _resolve_location(_build_filename(filename))
        path.write_text(content, encoding= "utf-8")
        print(f"[File Saver] Salvato in {path}")
        return {"final_response": f"{content}\n\n[File salvato in {path}]"}
    except Exception as e:
        print(f"[File Saver] Salvataggio non riuscito: {e}")
        return {"final_response": f"{content}\n\n[Salvataggio su file non riuscito: {e}]"}