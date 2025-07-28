from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Note: In a real app, embedding would be a separate, one-time process per document.
def get_relevant_clauses(question: str, index_name: str) -> str:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = PineconeVectorStore.from_existing_index(
        index_name=index_name,
        embedding=embeddings
    )
    retrieved_docs = vector_store.similarity_search(question, k=3)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    return context