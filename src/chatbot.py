# src/chatbot.py

import pandas as pd
from langchain.docstore.document import Document
from typing import List
from sentence_transformers import SentenceTransformer
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


class RAGChatbot:
    def __init__(
        self,
        csv_path: str,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        llm_model_name: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        top_k: int = 3,
    ):
        # Load and prepare data
        self.df = self.load_csv(csv_path)
        self.df = self.clean_data(self.df)
        self.documents = self.convert_to_documents(self.df)

        # Setup embedding model and vector store (FAISS)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.build_vector_index()

        # Setup local LLM
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(llm_model_name).to(device)
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if device == "cuda" else -1,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7,
        )

        self.top_k = top_k

    def load_csv(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path)
        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.fillna("Unknown")
        return df

    def convert_to_documents(self, df: pd.DataFrame) -> List[Document]:
        documents = []
        for i, row in df.iterrows():
            content = (
                f"Loan ID: {row['Loan_ID']}\n"
                f"Gender: {row['Gender']}, Married: {row['Married']}, Dependents: {row['Dependents']}, "
                f"Education: {row['Education']}, Self Employed: {row['Self_Employed']}\n"
                f"Applicant Income: {row['ApplicantIncome']}, Coapplicant Income: {row['CoapplicantIncome']}, "
                f"Loan Amount: {row['LoanAmount']}, Loan Term: {row['Loan_Amount_Term']}\n"
                f"Credit History: {row['Credit_History']}, Property Area: {row['Property_Area']}\n"
                f"Loan Status: {row['Loan_Status']}"
            )
            metadata = {"row": i, "Loan_ID": row.get("Loan_ID", "Unknown")}
            documents.append(Document(page_content=content, metadata=metadata))
        return documents

    def build_vector_index(self):
        texts = [doc.page_content for doc in self.documents]
        embeddings = self.embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        self.embeddings = embeddings  # keep for reference

    def similarity_search(self, query: str, top_k: int = None) -> List[Document]:
        if top_k is None:
            top_k = self.top_k
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        distances, indices = self.index.search(query_embedding, top_k)
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.documents):
                results.append(self.documents[idx])
        return results

    def generate_response(self, context: str, question: str) -> str:
        prompt = (
            "You are a helpful assistant. Use the information provided to answer the question.\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"USER QUESTION: {question}\n\n"
            "ANSWER:"
        )
        outputs = self.generator(prompt, max_new_tokens=128, clean_up_tokenization_spaces=True)
        full_text = outputs[0]['generated_text']
        # Extract answer only (remove prompt)
        answer = full_text[len(prompt):].strip()
        return answer

    def ask(self, question: str) -> dict:
        relevant_docs = self.similarity_search(question)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        answer = self.generate_response(context, question)
        return {
            "answer": answer,
            "sources": [doc.metadata for doc in relevant_docs]
        }


# For standalone testing
if __name__ == "__main__":
    import os
    csv_file = os.path.join(os.path.dirname(__file__), "..", "data", "Training_Dataset.csv")
    chatbot = RAGChatbot(csv_file)

    print("✅ Loaded and indexed documents:", len(chatbot.documents))
    test_question = "What are common characteristics of approved loans for married applicants?"
    print("\nQuestion:", test_question)
    response = chatbot.ask(test_question)
    print("\nAnswer:\n", response["answer"])
    print("\nSources:")
    for source in response["sources"]:
        print(source)
