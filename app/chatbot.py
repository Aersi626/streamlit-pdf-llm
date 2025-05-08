from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from app.retriever import get_retriever



def create_chatbot(vectorstore, llm_model="llama3.1"):
    llm = OllamaLLM(model=llm_model)

    # Define custom system prompt
    system_prompt = """You are an expert assistant for reading PDF documents.
    You may encounter tables in markdown format.
    If tables are provided in the context, read them carefully and use the table values to answer questions.
    If the answer is not in the provided context, say "I don't know based on the document.".
    """

    # Define the full prompt template
    qa_prompt = PromptTemplate.from_template(
        system_prompt + "\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
    )

    # Build combine_docs_chain, must be explicitly passed in LangChain >= 0.2.x.
    combine_docs_chain = StuffDocumentsChain(
        llm_chain=LLMChain(llm=llm, prompt=qa_prompt),
        document_variable_name="context"
    )

    # Question Generator Prompt (rephrasing), must be explicitly passed in LangChain >= 0.2.x.
    question_gen_prompt = PromptTemplate.from_template(
    """Given the following conversation history and a follow-up question, rephrase the follow-up question to be a standalone question.

    Chat History:
    {chat_history}
    Follow-up Question:
    {question}

    Standalone question:"""
    )

    question_generator = LLMChain(llm=llm, prompt=question_gen_prompt)

    # Create chain with custom prompt
    qa_chain = ConversationalRetrievalChain(
        retriever = get_retriever(vectorstore, k=1000),
        combine_docs_chain=combine_docs_chain,
        question_generator=question_generator,
        return_source_documents=True,
    )
    return qa_chain