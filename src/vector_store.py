# src/vector_store.py

import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

class VectorStore:
    def __init__(self, embedding_model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.documents = []
        self.embeddings = None

    def build_index(self, documents):
        """Embeds all documents and creates a FAISS index."""
        self.documents = documents
        texts = [doc.page_content for doc in documents]
        self.embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(self.embeddings)
        print(f"Built FAISS index with {len(self.documents)} documents.")

    def similarity_search(self, query, top_k=3):
        """Given a query string, return top_k most similar documents."""
        query_vec = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_vec, top_k)  # D: distances, I: indices
        results = []
        for idx in I[0]:
            if 0 <= idx < len(self.documents):
                results.append(self.documents[idx])
        return results

# --- Example usage ---
if __name__ == "__main__":
    from document_processor import load_and_process_csv
    from pathlib import Path

    csv_file = Path(__file__).parent.parent / "data" / "Training_Dataset.csv"
    docs = load_and_process_csv(csv_file)
    vector_store = VectorStore()
    vector_store.build_index(docs)
    query = "approved loans for married applicants"
    results = vector_store.similarity_search(query, top_k=3)
    print("\nTop results for query:", query)
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(doc.page_content)
