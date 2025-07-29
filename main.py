import os
import logging
import asyncio
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import groq # Import the groq library to catch its specific errors

# Import your utility functions and necessary clients
from utils.embedder import get_relevant_clauses
from utils.query_logic import evaluate_query
from langchain_groq import ChatGroq
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# --- Initialize Singleton Clients (Done once on startup) ---
try:
    logger.info("Initializing API clients...")
    llm_client = ChatGroq(
        temperature=0,
        model_name="llama3-70b-8192",
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
    logger.error(f"FATAL: Failed to initialize clients: {e}", exc_info=True)
    raise

# --- FastAPI App and Security ---
app = FastAPI(title="Policy Q&A API")
API_KEY = os.getenv("HACKRX_TEAM_TOKEN")
api_key_header = APIKeyHeader(name="Authorization")

def get_api_key(api_key_from_header: str = Security(api_key_header)):
    if api_key_from_header == f"Bearer {API_KEY}":
        return api_key_from_header
    raise HTTPException(status_code=401, detail="Invalid or Missing API Key")

# --- Pydantic Models ---
class QueryInput(BaseModel):
    documents: str
    questions: List[str]

class QueryOutput(BaseModel):
    answers: List[str]

# --- Concurrency and Processing Logic ---
# Reduced concurrency to lower the request rate
CONCURRENT_REQUESTS_LIMIT = 2
semaphore = asyncio.Semaphore(CONCURRENT_REQUESTS_LIMIT)
MAX_RETRIES = 3
RETRY_DELAY_SECONDS = 15 # Wait 15 seconds between retries

async def process_single_question(question: str) -> str:
    """
    Processes a single question with retries for rate limit errors.
    """
    async with semaphore:
        for attempt in range(MAX_RETRIES):
            try:
                loop = asyncio.get_running_loop()
                context = await loop.run_in_executor(
                    None, get_relevant_clauses, question, vector_store
                )
                final_answer = await loop.run_in_executor(
                    None, evaluate_query, question, context, llm_client
                )
                return final_answer
            except groq.RateLimitError as e:
                logger.warning(
                    f"Rate limit hit for question '{question}' on attempt {attempt + 1}. "
                    f"Retrying in {RETRY_DELAY_SECONDS}s."
                )
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(RETRY_DELAY_SECONDS)
                else:
                    logger.error(f"Max retries reached for question '{question}'. Failing. Error: {e}")
                    return "The API is currently busy. Please try again in a moment."
            except Exception as e:
                logger.error(f"An unexpected error occurred processing question '{question}': {e}", exc_info=True)
                return "An error occurred while processing this question."

# --- API Endpoints ---
@app.get("/", tags=["Health Check"])
async def read_root():
    """Root endpoint for health checks and service availability."""
    return {"status": "ok", "message": "Welcome to the Policy Q&A API!"}

@app.post("/api/v1/hackrx/run", response_model=QueryOutput, tags=["Q&A"])
async def run_submission(input_data: QueryInput, api_key: str = Security(get_api_key)):
    """Receives questions and returns AI-generated answers based on policy documents."""
    try:
        tasks = [process_single_question(q) for q in input_data.questions]
        answers = await asyncio.gather(*tasks)
        return {"answers": answers}
    except Exception as e:
        logger.error(f"An unexpected error occurred during API call: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")
