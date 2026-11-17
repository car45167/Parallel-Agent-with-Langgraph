from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_huggingface import HuggingFaceEmbeddings
from pathlib import Path


## Altrimenti non legge la gpu amd. Verrà introdotta una fallback in un secondo momento
os.environ["HIP_VISIBLE_DEVICES"] = "1"

def carica_documenti(directory):
    document_list= []
    for pdf in Path(directory).rglob("*.pdf"):
        for doc in PyMuPDFLoader(str(pdf)).load():
            doc.metadata["source"] = pdf.name
            document_list.append(doc)
    return document_list

CHROMA_DB = Path.cwd().resolve()/ "chroma_db" ## Workspace/chroma_db/...

def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name= "BAAI/bge-m3",
        model_kwargs={"device":"cuda"},
        encode_kwargs={"batch_size":16}
    )


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

    output_dir = input("Inserire un nome per la collection: ") ## Vincolo al momento non accetta spazi o altri caratteri speciali come ' " o !

    ##Creazione del db
    try: 
        Chroma.from_documents(documents=chunks,embedding=embeddings,
                              persist_directory=CHROMA_DB,
                              collection_name=str(output_dir))
        print(f"[SUCCESS]Creazione del Database eseguita con successo\n")
    except Exception as e:
        print(f"[FAILED] Creazione del database fallita: {e}")


if __name__ =="__main__":
    main()

