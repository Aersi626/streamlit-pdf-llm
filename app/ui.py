import streamlit as st
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.pdf_loader import load_and_split_pdf
from app.vector_store import build_vectorstore, load_vectorstore, get_index_dir, index_exists
from app.chatbot import create_chatbot

PDF_DIR = "./data/pdfs"
INDEX_DIR = "./models/faiss_index"

st.set_page_config(page_title="PDF RAG Chatbot")
st.title("📄 PDF Chatbot (Llama 3.1 + RAG)")
st.write("Upload a PDF and chat with it using Retrieval-Augmented Generation.")

os.makedirs(PDF_DIR, exist_ok=True)

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Load or build vectorstore
if uploaded_file:
    file_path = os.path.join(PDF_DIR, uploaded_file.name)
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    index_dir = get_index_dir(file_path)

    if "vectorstore" not in st.session_state or st.session_state.get("current_pdf") != uploaded_file.name:
        if index_exists(index_dir):
            st.success("Index found for this PDF, loading...")
            vectorstore = load_vectorstore(index_dir)
        else:
            with st.spinner("Processing and indexing PDF..."):
                documents = load_and_split_pdf(file_path)
                vectorstore = build_vectorstore(documents, save_path=INDEX_DIR)
        st.session_state.vectorstore = vectorstore
        st.session_state.current_pdf = uploaded_file.name
    else:
        vectorstore = st.session_state.vectorstore
else:
    st.warning("Please upload a PDF to start.")
    st.stop()

if "chatbot" not in st.session_state or st.session_state.get("current_pdf") != uploaded_file.name:
    st.session_state.chatbot = create_chatbot(vectorstore)

chatbot = st.session_state.chatbot

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Ask a question about the PDF:")

if user_input:
    result = chatbot.invoke({"question": user_input,
                      "chat_history": st.session_state.chat_history})

    st.session_state.chat_history.append(("You", user_input))
    st.session_state.chat_history.append(("Bot", result["answer"]))

# Display chat history
for speaker, msg in st.session_state.chat_history:
    st.markdown(f"**{speaker}:** {msg}")