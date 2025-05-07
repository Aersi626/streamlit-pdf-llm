from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.schema import Document
import os
import streamlit as st
import hashlib
import faiss

def get_index_dir(pdf_file_path):
    filename = os.path.basename(pdf_file_path)
    file_hash = hashlib.md5(filename.encode()).hexdigest()
    return f"./models/faiss_index/{file_hash}"

def index_exists(index_dir):
    return os.path.exists(os.path.join(index_dir, "index.faiss"))

def embed_text(embedding_model, text):
    embedder = OllamaEmbeddings(model=embedding_model)
    return embedder.embed_query(text)

def build_vectorstore(documents, embedding_model="nomic-embed-text", save_path="./models/faiss_index"):
    embeddings = OllamaEmbeddings(model=embedding_model)

    total = len(documents)
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Prepare texts
    texts = [doc.page_content for doc in documents]
    metadatas = [doc.metadata for doc in documents]

    embedded_documents = []
    embeddings_list = [None] * total

    # Parallel embedding with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_index = {
            executor.submit(embed_text, embedding_model, text): idx
            for idx, text in enumerate(texts)
        }

        for count, future in enumerate(as_completed(future_to_index)):
            idx = future_to_index[future]
            embed = future.result()
            embeddings_list[idx] = embed

            progress_bar.progress((count + 1) / total)
            status_text.text(f"Embedding chunk {count + 1} of {total}")

    # Build Documents with embeddings in metadata
    for i, text in enumerate(texts):
        doc = Document(page_content=text, metadata=metadatas[i])
        embedded_documents.append(doc)

    faiss.omp_set_num_threads(16)
    status_text.text("Building FAISS index, please wait...")
    vectorstore = FAISS.from_documents(embedded_documents, embeddings)

    status_text.text("Saving to disk, please wait...")
    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)

    progress_bar.empty()
    status_text.empty()

    return vectorstore

def load_vectorstore(load_path="./models/faiss_index", embedding_model="nomic-embed-text"):
    embeddings = OllamaEmbeddings(model=embedding_model)
    return FAISS.load_local(load_path, embeddings, allow_dangerous_deserialization=True)