from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore

# --- Embed Document ---
def embed_document(text: str, index_name: str, embeddings_client: HuggingFaceEndpointEmbeddings):
    """Splits text, creates embeddings, and stores them in Pinecone using a provided client."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)

    print(f"Embedding {len(chunks)} chunks into Pinecone index '{index_name}'...")
    PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings_client,
        index_name=index_name
    )
    print("Embedding complete.")

# --- Retrieve Relevant Clauses ---
def get_relevant_clauses(question: str, vector_store: PineconeVectorStore, top_k: int = 4) -> str:
    """Finds and returns the most relevant text chunks using a provided vector store."""
    retrieved_docs = vector_store.similarity_search(question, k=top_k)
    return "\n\n".join([doc.page_content for doc in retrieved_docs])