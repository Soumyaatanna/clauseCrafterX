from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

def evaluate_query(question: str, context: str) -> str:
    prompt_template = """
    You are an expert at analyzing policy documents.
    Answer the user's question based ONLY on the following context.
    If the context does not contain the answer, state that the information is not available in the provided text.

    CONTEXT:
    {context}

    QUESTION:
    {question}

    ANSWER:
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    chain = prompt | llm | StrOutputParser()
    final_answer = chain.invoke({"context": context, "question": question})
    return final_answer