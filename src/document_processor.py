# src/document_processor.py

import pandas as pd
from pathlib import Path

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

def load_and_process_csv(csv_path):
    df = pd.read_csv(csv_path)
    # Fill NaNs for cleaner formatting
    df = df.fillna("N/A")
    documents = []
    for idx, row in df.iterrows():
        content = (
            f"Loan Application ID: {row.get('Loan_ID', 'N/A')}\n"
            f"Gender: {row.get('Gender', 'N/A')}\n"
            f"Married: {row.get('Married', 'N/A')}\n"
            f"Dependents: {row.get('Dependents', 'N/A')}\n"
            f"Education: {row.get('Education', 'N/A')}\n"
            f"Self_Employed: {row.get('Self_Employed', 'N/A')}\n"
            f"ApplicantIncome: {row.get('ApplicantIncome', 'N/A')}\n"
            f"CoapplicantIncome: {row.get('CoapplicantIncome', 'N/A')}\n"
            f"LoanAmount: {row.get('LoanAmount', 'N/A')}\n"
            f"Loan_Amount_Term: {row.get('Loan_Amount_Term', 'N/A')}\n"
            f"Credit_History: {row.get('Credit_History', 'N/A')}\n"
            f"Property_Area: {row.get('Property_Area', 'N/A')}\n"
            f"Loan_Status: {row.get('Loan_Status', 'N/A')}\n"
        )
        metadata = {"row_id": idx, "Loan_ID": row.get("Loan_ID", "N/A")}
        documents.append(Document(content, metadata))
    return documents

# Example use
if __name__ == "__main__":
    csv_file = Path(__file__).parent.parent / "data" / "Training_Dataset.csv"
    docs = load_and_process_csv(csv_file)
    print(f"Loaded {len(docs)} documents. Example document:\n")
    print("---")
    print(docs[0].page_content)
    print("Metadata:", docs[0].metadata)
