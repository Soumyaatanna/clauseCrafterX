import os
import logging
import asyncio
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv

# Import refactored functions and necessary clients
from utils.embedder import get_relevant_clauses
from utils.query_logic import evaluate_query
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_pinecone import PineconeVectorStore

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# --- Initialize Singleton Clients ---
# These clients are created once and reused for all requests.
try:
    llm_client = ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",  # Upgraded model for better accuracy
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    embeddings_client = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-small-en-v1.5",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )

    pinecone_index_name = os.getenv("PINECONE_INDEX")
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=pinecone_index_name,
        embedding=embeddings_client
    )
    
    logger.info("All clients initialized successfully.")

except Exception as e:
    logger.error(f"Failed to initialize clients: {e}", exc_info=True)
    # If clients fail to init, the app shouldn't start. You can handle this more gracefully.
    raise

# --- FastAPI App and Security ---
app = FastAPI()
API_KEY = os.getenv("HACKRX_TEAM_TOKEN")
api_key_header = APIKeyHeader(name="Authorization")

def get_api_key(api_key_from_header: str = Security(api_key_header)):
    if api_key_from_header == f"Bearer {API_KEY}":
        return api_key_from_header
    raise HTTPException(status_code=401, detail="Invalid or Missing API Key")

# --- Pydantic Models ---
class QueryInput(BaseModel):
    documents: str  # Kept for compatibility with the expected input schema
    questions: List[str]

class QueryOutput(BaseModel):
    answers: List[str]

# --- Throttle control ---
semaphore = asyncio.Semaphore(5)  # Increased slightly as client init is no longer a bottleneck

async def process_single_question(question: str) -> str:
    """Processes a single question using the pre-initialized clients."""
    # The vector_store and llm_client are now accessed from the global scope.
    context = get_relevant_clauses(question, vector_store)
    final_answer = evaluate_query(question, context, llm_client)
    return final_answer

async def throttled_process(question: str) -> str:
    """Throttles requests to avoid exceeding external API rate limits."""
    async with semaphore:
        loop = asyncio.get_running_loop()
        # Use run_in_executor for the blocking I/O calls within the async function
        return await loop.run_in_executor(None, process_single_question, question)

# --- Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=QueryOutput)
async def run_submission(input_data: QueryInput, api_key: str = Security(get_api_key)):
    try:
        tasks = [process_single_question(q) for q in input_data.questions]
        answers = await asyncio.gather(*tasks)
        return {"answers": answers}
    except Exception as e:
        logger.error("An unexpected error occurred during API call", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")