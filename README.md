# 🧠 Policy Q&A AI Assistant

This is a powerful AI-based API designed to answer questions from any policy document using an LLM-powered RAG (Retrieval-Augmented Generation) pipeline. It processes the input document (PDF, DOCX, or TXT), creates vector embeddings, and uses them to retrieve the most relevant chunks that are passed to Groq’s blazing-fast LLM (Llama 3 70B) for intelligent answers.

---

## ✅ Features

- **Document Processing:** Extracts content from PDF, DOCX, and TXT files hosted online.
- **Vector Embeddings:** Embeds text using Hugging Face models.
- **Semantic Search:** Uses Pinecone vector DB to fetch top relevant content.
- **Answer Generation:** Queries are processed using Groq-hosted Llama 3 70B for fast and accurate answers.
- **Asynchronous API:** Built with FastAPI and supports concurrent requests.
- **API Key Security:** Endpoint is secured with token-based authentication.
- **Resilient:** Includes retry logic with exponential backoff to handle rate limits.

---

## 🛠 Tech Stack

- **Backend:** Python, FastAPI  
- **LLM & Embeddings:** Groq (Llama 3 70B), Hugging Face, LangChain  
- **Vector Store:** Pinecone  
- **Deployment:** Gunicorn, Uvicorn, Render  
- **Libraries:** `pydantic`, `python-dotenv`, `httpx`, `langchain`

---

## ⚙️ Setup & Installation

### 1. Clone the Repo
```bash
git clone https://github.com/Soumyaatanna/clauseCrafterX.git
cd clauseCrafterX
2. Create Virtual Environment
bash
Copy
Edit
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```
3. Install Dependencies
```
bash
Copy
Edit
pip install -r requirements.txt
```
5. Add Environment Variables
Create a .env file in the root directory:
```
env
Copy
Edit
GROQ_API_KEY="gsk_..."
HUGGINGFACEHUB_API_TOKEN="hf_..."
PINECONE_API_KEY="pcsk_..."
PINECONE_ENV="gcp-starter"
PINECONE_INDEX="hackrx-index"
HACKRX_TEAM_TOKEN="your_strong_secret_token_here"
```
🧾 One-Time Setup: Document Processing
Before starting the server, process your document and upload embeddings:
```
bash
Copy
Edit
python setup_document.py
```
This script will:

Download the document

Extract text

Split text into chunks

Generate embeddings

Upload to Pinecone

🚀 Start the API Server
```
bash
Copy
Edit
uvicorn main:app --reload
Server will be live at: http://127.0.0.1:8000
```
🔐 API Usage
```
Endpoint:
POST /api/v1/hackrx/run
```
Headers
Key	Value
Content-Type	application/json
Authorization	Bearer <your_token>

Request Body
```
json
Copy
Edit
{
  "documents": "policy",
  "questions": [
    "What is the waiting period for pre-existing diseases?",
    "Are maternity expenses covered?",
    "What is the No Claim Discount offered?"
  ]
}
```
Sample curl Request
```
bash
Copy
Edit
curl -X POST "http://127.0.0.1:8000/api/v1/hackrx/run" \
-H "Content-Type: application/json" \
-H "Authorization: Bearer <your_token>" \
-d '{
  "documents": "policy",
  "questions": ["What is the waiting period for cataract surgery?"]
}'
Sample Response
json
Copy
Edit
{
  "answers": [
    "The waiting period for cataract surgery is 2 years..."
  ]
}
```
☁️ Deployment (Render)
Push the code to GitHub

Create a new Web Service on Render

Set:

Runtime: Python 3

Build Command: pip install -r requirements.txt

Start Command:
```
bash
Copy
Edit
gunicorn -w 4 -k uvicorn.workers.UvicornWorker main:app
Add environment variables from .env
```
Deploy. Done ✅

Note: Run the document setup (setup_document.py) locally before deploying.

🤝 Contributing
Feel free to open issues or PRs. For improvements, bug fixes, or integrations — all contributions are welcome.

📄 License
MIT License (add a LICENSE file if not already included).

🙋 Contact
Maintained by Soumya Tanna
