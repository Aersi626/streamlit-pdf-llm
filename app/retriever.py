from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever
from typing import Optional, Dict, Any, List
import re

class PageTableAwareRetriever(VectorStoreRetriever):
    def __init__(self, vectorstore: FAISS, k: int = 12):
        super().__init__(vectorstore=vectorstore, search_kwargs={"k": k})
        self.vectorstore = vectorstore

    def extract_page_hint(self, query: str) -> Optional[int]:
        # Extract page number if user asks "page 71" or "on page 71"
        page_match = re.search(r'page (\d+)', query, re.IGNORECASE)
        if page_match:
            return int(page_match.group(1))
        return None

    def extract_table_hint(self, query: str) -> Optional[str]:
        # Extract table number like "Table 8.2" or "table 5"
        table_match = re.search(r'table (\d+(\.\d+)?)', query, re.IGNORECASE)
        if table_match:
            return table_match.group(1)
        return None

    def get_relevant_documents(self, query: str) -> List[Document]:
        page_hint = self.extract_page_hint(query)
        table_hint = self.extract_table_hint(query)

        k = self.search_kwargs.get("k", 12)
        docs = self.vectorstore.similarity_search(query, k=k)

        boosted_docs = []

        for doc in docs:
            score = 0
            metadata = doc.metadata

            # Boost page matches
            if page_hint and "page" in metadata and metadata["page"] == page_hint:
                score += 2

            # Boost table matches
            if table_hint and "page_content" in doc.__dict__ and f"Table {table_hint}" in doc.page_content:
                score += 1

            boosted_docs.append((score, doc))

        if page_hint:
            # Retrieve ALL documents (be careful: assumes vectorstore is not huge)
            all_docs = self.vectorstore.similarity_search("", k=9999)  # blank query returns all
            page_docs = [doc for doc in all_docs if doc.metadata.get("page") == page_hint]

            # Avoid duplicates
            existing_doc_ids = {id(doc) for _, doc in boosted_docs}
            for pd in page_docs:
                if id(pd) not in existing_doc_ids:
                    boosted_docs.append((5, pd))  # Strong boost → force page

        # Sort by boost score first, then keep original order
        boosted_docs = sorted(boosted_docs, key=lambda x: -x[0])

        # Return only documents (drop scores)
        return [doc for _, doc in boosted_docs]

def get_retriever(vectorstore: FAISS, k: int = 12) -> VectorStoreRetriever:
    """
    Factory function to get the page + table aware retriever
    """
    return PageTableAwareRetriever(vectorstore=vectorstore, k=k)
