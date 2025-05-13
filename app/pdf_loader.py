from langchain.schema import Document
from app.table_extractor import extract_tables_from_pdf, extract_text_without_tables
from app.table_formatter import format_table_as_markdown_json_hybrid

import pdfplumber
import os
import pandas as pd

def chunk_table(df: pd.DataFrame, page_num: int, table_num: int, table_caption: str, chunk_size: int = 200) -> list[str]:
    """
    Splits a DataFrame into markdown chunks with repeated headers.
    """
    chunks = []
    total_rows = len(df)

    for start in range(0, total_rows, chunk_size):
        chunk_df = df.iloc[start:start + chunk_size].reset_index(drop=True)
        chunk_text = format_table_as_markdown_json_hybrid(chunk_df, page_num, table_num, table_caption)
        chunks.append(chunk_text)

    return chunks

def load_and_split_pdf(pdf_path: str) -> list[Document]:
    """
    Loads text and tables from a PDF, returns as LangChain Documents.
    Saves extracted content to separate .txt/.md logs for debugging or context injection.
    """
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    documents = []

    # Load non-table text using pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = extract_text_without_tables(page)
            if text:
                documents.append(Document(
                    page_content=text,
                    metadata={"source": base_name, "page": page_num, "type": "text"}
                ))

    # Add chunked tables as separate documents
    tables = extract_tables_from_pdf(pdf_path)
    for df, page_num, table_num, table_caption in tables:
        table_chunks = chunk_table(df, page_num=page_num, table_num=table_num, table_caption=table_caption, chunk_size=200)
        for chunk_idx, chunk_text in enumerate(table_chunks, start=1):
            documents.append(Document(
                page_content=chunk_text,
                metadata={
                    "source": base_name,
                    "page": page_num,
                    "table": table_num,
                    "chunk": chunk_idx,
                    "caption": table_caption,
                    "type": "table"
                }
            ))

    # Save extracted text and tables to separate files for debugging
    log_path = os.path.join("logs", f"{base_name}_rag_context.txt")
    log_documents_for_debugging(documents, log_path)

    return documents

def log_documents_for_debugging(docs: list, output_path: str):
    """Write LangChain Document contents to a .txt file for visibility."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, doc in enumerate(docs, 1):
            f.write(f"\n\n===== Document {i} =====\n")
            f.write(f"Metadata: {doc.metadata}\n")
            f.write(f"\nContent:\n{doc.page_content[:10000]}")  # Trim for readability
