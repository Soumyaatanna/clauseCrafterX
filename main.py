import os
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
from utils.embedder import get_relevant_clauses
from utils.query_logic import evaluate_query

# Load environment variables from .env file
load_dotenv()

app = FastAPI()

# --- Security ---
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

# --- API Endpoint ---
@app.post("/api/v1/hackrx/run", response_model=QueryOutput)
async def run_submission(input_data: QueryInput, api_key: str = Security(get_api_key)):
    try:
        pinecone_index_name = os.getenv("PINECONE_INDEX")
        answers = []
        for question in input_data.questions:
            context = get_relevant_clauses(question, pinecone_index_name)
            final_answer = evaluate_query(question, context)
            answers.append(final_answer)
        return {"answers": answers}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))