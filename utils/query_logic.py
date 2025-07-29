from langchain_google_genai import ChatGoogleGenerativeAI

from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

def evaluate_query(question: str, context: str) -> str:
    prompt_template = """
    You are an expert assistant who answers questions about a policy document.
Answer the user's question directly and concisely using only the facts from the provided context.
If the information to answer the question is not in the context, state that the information is not available in the policy.


    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
    chain = prompt | llm | StrOutputParser()
    final_answer = chain.invoke({"context": context, "question": question})
    return final_answer