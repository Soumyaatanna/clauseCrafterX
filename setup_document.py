# In setup_document.py

import os
from dotenv import load_dotenv
from utils.parser import extract_text_from_url
from utils.embedder import embed_document

# Load environment variables
load_dotenv()

# URL of the document to process
DOCUMENT_URL = "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D"
PINECONE_INDEX = os.getenv("PINECONE_INDEX")

if __name__ == '__main__':
    print("Starting document processing...")
    # 1. Extract text from the document URL
    full_text = extract_text_from_url(DOCUMENT_URL)
    
    # 2. Embed the text and store it in Pinecone
    embed_document(full_text, PINECONE_INDEX)
    
    print("Setup complete. Your index is ready.")