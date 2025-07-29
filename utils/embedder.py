from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# ADD THIS FUNCTION
def embed_document(text: str, index_name: str):
    """Splits text, creates embeddings, and stores them in Pinecone."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    print(f"Embedding {len(chunks)} chunks into Pinecone index '{index_name}'...")
    # This creates the vector store and upserts the data
    PineconeVectorStore.from_texts(
        texts=chunks, 
        embedding=embeddings, 
        index_name=index_name
    )
    print("Embedding complete.")


def get_relevant_clauses(question: str, index_name: str) -> str:
    """Finds and returns the most relevant text chunks for a given question."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=index_name, 
        embedding=embeddings
    )
    retrieved_docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return context