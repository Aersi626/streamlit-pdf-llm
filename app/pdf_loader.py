import pdfplumber
from langchain.schema import Document
from typing import List

def table_to_markdown(table: List[List[str]]) -> str:
    if not table or len(table) < 2:
        return ""

    # Safe row (None -> "")
    def safe_row(row):
        return [str(cell).strip() if cell is not None else "" for cell in row]

    header = safe_row(table[0])
    rows = [safe_row(row) for row in table[1:]]

    # Build markdown
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"] * len(header)) + " |\n"

    for row in rows:
        md += "| " + " | ".join(row) + " |\n"

    return md

def load_and_split_pdf(file_path: str) -> List[Document]:
    documents = []

    with pdfplumber.open(file_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""

            tables = page.extract_tables()

            # Add page text if exists
            if text.strip():
                documents.append(Document(page_content=f"Page {page_num}:\n\n{text}",
                                          metadata={"page": page_num, "type": "text"}))

            # Add each table as separate document
            for table_num, table in enumerate(tables, start=1):
                if table:
                    table_md = f"### Table {table_num} on Page {page_num}\n\n"
                    table_md += table_to_markdown(table)

                    documents.append(Document(page_content=table_md,
                                              metadata={"page": page_num, "type": "table", "table_number": table_num}))

    return documents
