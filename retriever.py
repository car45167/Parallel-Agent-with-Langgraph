
import torch
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from config import COLLECTION_NAME,CHROMA_DB
import os
from functools import lru_cache
## Altrimenti non mi legge AMD
os.environ["HIP_VISIBLE_DEVICES"] = "1"

torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)


def _get_device(): 
    if torch.cuda.is_available():
        print("[CUDA]")
        return torch.device("cuda"), 64
    else:
        print("[CPU]")
        return torch.device("cpu"), 8


@lru_cache(maxsize = 1) ## Caricamento lazy in cache del get_embeddings altrimenti ad ogni start devo caricare il modello in gpu. Se non si usa RAG, non carica nulla.
def get_embeddings():
    device , size= _get_device()
    return HuggingFaceEmbeddings(
        model_name= "BAAI/bge-m3",
        model_kwargs={"device":device},
        encode_kwargs={"batch_size":size}
    )


def _get_vector_db(collection:str = COLLECTION_NAME)-> Chroma:
    return Chroma(collection_name=collection, 
                  persist_directory=str(CHROMA_DB),
                  embedding_function=get_embeddings())


def retrieve(query:str, k: int=4):
    vector_db=_get_vector_db(COLLECTION_NAME).as_retriever(search_kwargs={"k":k})
    result = vector_db.invoke(query)
    return result