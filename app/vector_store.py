from langchain_community.vectorstores import FAISS
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.schema import Document
from app.embedding import BGE_Embedding
from langchain.text_splitter import MarkdownTextSplitter
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

def build_vectorstore(documents, embedding_model="BAAI/bge-large-en", save_path="./models/faiss_index"):
    splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = splitter.split_documents(documents)
    
    embedder = BGE_Embedding(model_name=embedding_model)

    total = len(docs)
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Prepare texts
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]

    embeddings_list = [None] * total

    # Parallel embedding with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=12) as executor:
        future_to_index = {
            executor.submit(embedder.embed_text, text): idx
            for idx, text in enumerate(texts)
        }

        for count, future in enumerate(as_completed(future_to_index)):
            idx = future_to_index[future]
            embed = future.result()

            if embed is None or not isinstance(embed, list):
                raise ValueError(f"Embedding failed for index {idx} with result {embed}")
            
            embeddings_list[idx] = embed

            progress_bar.progress((count + 1) / total)
            status_text.text(f"Embedding chunk {count + 1} of {total}")

    text_embeddings = list(zip(texts, embeddings_list))

    faiss.omp_set_num_threads(12)
    status_text.text("Building FAISS index, please wait...")
    vectorstore = FAISS.from_embeddings(text_embeddings, embedding=embedder, metadatas=metadatas)

    status_text.text("Saving to disk, please wait...")
    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(save_path)

    progress_bar.empty()
    status_text.empty()

    return vectorstore

def load_vectorstore(load_path="./models/faiss_index", embedding_model="BAAI/bge-large-en"):
    embedder = BGE_Embedding(model=embedding_model)
    return FAISS.load_local(load_path, embedder, allow_dangerous_deserialization=True)