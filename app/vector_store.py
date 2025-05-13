from langchain_community.vectorstores import FAISS
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.schema import Document
from app.embedding import BGE_Embedding
from langchain.text_splitter import MarkdownTextSplitter, RecursiveCharacterTextSplitter
import os
import streamlit as st
import hashlib
import faiss
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def get_index_dir(pdf_file_path):
    filename = os.path.basename(pdf_file_path)
    file_hash = hashlib.md5(filename.encode()).hexdigest()
    return f"./models/faiss_index/{file_hash}"

def index_exists(index_dir):
    return os.path.exists(os.path.join(index_dir, "index.faiss"))

def build_vectorstore(documents, embedding_model="BAAI/bge-large-en", save_path="./models/faiss_index"):
    #splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=200)
    # improve performance by using RecursiveCharacterTextSplitter
    # with separators that are more likely to split at natural boundaries
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
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

    # DEBUG: Check a specific chunk's embedding
    print("🔍 Sample text chunk:\n", texts[0][:300])
    print("🔢 Sample embedding (truncated):", embeddings_list[0][:10])
    query = "do you see 60007 RAWDATA?"
    query_vec = embedder.embed_query(query)
    print("❓ Query embedding (truncated):", query_vec[:10])

    save_embeddings(texts, embeddings_list, f"./logs/embeddings.txt")

    # Assume embeddings_list is a list of list[float]
    X = np.array(embeddings_list)
    X_pca = PCA(n_components=2).fit_transform(X)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
    plt.title("Document Chunk Embeddings (PCA)")
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")

    # Save the plot
    output_path = "./logs/embedding_pca_plot.png"
    plt.savefig(output_path)

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

def save_embeddings(texts: list[str], vectors: list[list[float]], path: str):
    """Save texts and their embeddings side-by-side."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for text, vec in zip(texts, vectors):
            f.write("\n\n===== Text Chunk =====\n")
            f.write(text[:500].replace("\n", " ") + "\n")
            f.write("Vector (truncated):\n")
            f.write(np.array(vec[:10]).round(3).astype(str).tolist().__str__() + "...\n")