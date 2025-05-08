from sentence_transformers import SentenceTransformer
from typing import List
import torch

class BGE_Embedding:
    def __init__(self, model_name="hkunlp/instructor-xl"):
        self.model = SentenceTransformer(model_name, device="cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        if "instructor" in model_name.lower():
            self.instruction = "Represent the document for retrieval:"
        else:
            self.instruction = None

    def __call__(self, text: str) -> list[float]:
        return self.embed_query(text)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.instruction:
            inputs = [[self.instruction, text] for text in texts]
        else:
            inputs = texts

        return self.model.encode(inputs, normalize_embeddings=True).tolist()
    
    def embed_text(self, text: str) -> List[float]:
        if self.instruction:
            inputs = [[self.instruction, text]]
        else:
            inputs = [text]

        return self.model.encode(inputs, normalize_embeddings=True)[0].tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_texts(texts)
    
    def embed_query(self, text: str) -> List[float]:
        return self.embed_text(text)