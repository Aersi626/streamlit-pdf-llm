from langchain.chains import ConversationalRetrievalChain
from langchain_ollama import OllamaLLM

def create_chatbot(vectorstore, llm_model="llama3"):
    llm = OllamaLLM(model=llm_model)

    chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 8}),
        return_source_documents=True,
        output_key="answer",
        verbose=True
    )
    return chain