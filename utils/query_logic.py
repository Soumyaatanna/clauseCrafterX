from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

def evaluate_query(question: str, context: str, llm_client: ChatGroq) -> str:
    prompt_template = """
    You are a helpful and friendly policy expert AI assistant.
    Your goal is to answer the user's QUESTION using only the provided CONTEXT.
    Please follow these rules carefully:

    1.  **Analyze the CONTEXT:** Read the context thoroughly to find the information needed to answer the QUESTION.
    2.  **Be Accurate:** Base your answer strictly on the information given in the CONTEXT. Do not add any information or make assumptions.
    3.  **Use Clear Formatting:** Write in a clear and easy-to-understand manner. Use bullet points to list out criteria, exclusions, or steps.
    4.  **Answer Directly:** If the question can be answered with "Yes" or "No", please start your answer with that word, followed by the detailed explanation.
    5.  **Handle Missing Information:** If the answer cannot be found in the CONTEXT, you must respond with exactly this phrase: "The specific detail is not available in the provided information."

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    chain = prompt | llm_client | StrOutputParser()
    final_answer = chain.invoke({"context": context, "question": question})

    # Use .strip() to remove leading/trailing whitespace and newlines
    return final_answer.strip()