from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from retriever import get_embeddings
from pathlib import Path
from config import CHROMA_DB, COLLECTION_NAME

## Altrimenti non mi legge AMD
os.environ["HIP_VISIBLE_DEVICES"] = "1"
## Cartella di salvataggio del database è hardcoded nel workspace così come la funzione per il check della collection name sono scelte progettuali per limitare la possibilità di prompt injection quando gli llm leggeranno i nomi .

### PyMUPDF mantiene di per se la source dei dati ma in formato PATH assoluto. In questo caso prendo solo il nome del pdf.
def carica_documenti(directory):
    base_dir= Path(directory).resolve() ##Trasformazione di directory in Path Assoluto.
    document_list= []
    for pdf in base_dir.rglob("*.pdf"):
        for doc in PyMuPDFLoader(str(pdf)).load():
            doc.metadata["source"] = str(pdf.relative_to(base_dir)) ## Trasformazione di base_dir in path relativo per gestire eventuali sottocartelle e sources duplicate presenti in esse
            document_list.append(doc)
    return document_list




def check_cartella(directory):
    if Path(directory).is_dir():  ## Anche if os.path.exists()
        print(f"Cartella esistente")
        return True
    print("Cartella non esistente, inserisce una cartella esistente")
    return False


## Esecuzione singola dello script, che non viene richiamato. Di conseguenza definisco il main per poi richiamarlo.
def main():
    directory = input("Inserisci il path della directory contenente i file: \n")
    while not check_cartella(directory):
            directory= input("Inserisci il path della directory contenente i file: \n")
    raw_documents = carica_documenti(directory)
    print(f"Pagine totali: {len(raw_documents)}\n")

    ## Definizione dello splitter per la suddivisione in chunk
    chunks = RecursiveCharacterTextSplitter(chunk_size = 1500, chunk_overlap = 200).split_documents(raw_documents)
    print(f"Chunks totali: {len(chunks)}\n")

    ##Creazione degli embeddings
    embeddings = get_embeddings()


    ##Creazione del db
    try: 
        Chroma.from_documents(documents=chunks,embedding=embeddings,
                              persist_directory=CHROMA_DB,
                              collection_name=COLLECTION_NAME)
        print(f"[SUCCESS]Creazione del Database eseguita con successo\n")
    except Exception as e:
        print(f"[FAILED] Creazione del database fallita: {e}")


if __name__ =="__main__":
    main()

