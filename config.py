from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from pathlib import Path

MAX_TOOL_CALLS = 1

load_dotenv()
ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID")
CF_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN")

WORKSPACE = Path.cwd().resolve()

llm = ChatOpenAI(
    model="@cf/qwen/qwen3.8-27b",
    base_url=f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT_ID}/ai/v1",
    api_key=CF_TOKEN,
    max_tokens=4096,
    timeout= 60,   ## Tempo massimo di attesa per la risposta
)



CHROMA_DB = WORKSPACE / "chroma_db" ## Workspace/chroma_db/...