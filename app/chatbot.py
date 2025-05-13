from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from app.retriever import get_retriever
from langchain.callbacks import StdOutCallbackHandler



def create_chatbot(vectorstore, llm_model="llama3.1:8b-instruct-q8_0"):
    llm = OllamaLLM(model=llm_model)

    # Define custom system prompt
    system_prompt = """You are an expert assistant. The context includes Markdown tables.
    Pay attention to column headers and row values when answering questions.
    Only use the information from the provided context.".
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
        retriever = get_retriever(vectorstore, k=10),
        combine_docs_chain=combine_docs_chain,
        question_generator=question_generator,
        return_source_documents=True,
        verbose=True,
        callbacks=[StdOutCallbackHandler()]
    )

    retriever = get_retriever(vectorstore, k=10)
    docs = retriever.get_relevant_documents("do you see 60007 RAWDATA?")
    print("🔍 Retrieved documents for test query:")
    for i, doc in enumerate(docs):
        print(f"--- Doc {i+1} ---")
        print("Metadata:", doc.metadata)
        print(doc.page_content[:1000], "\n")

    return qa_chain