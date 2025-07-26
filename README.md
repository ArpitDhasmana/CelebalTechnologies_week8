```markdown
# RAG Q&A Chatbot for Loan Approval Prediction

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20App-blue?logo=streamlit)](YOUR_LIVE_STREAMLIT_APP_LINK_HERE)

---

## Overview

This project implements a **Retrieval-Augmented Generation (RAG) Chatbot** that provides intelligent and data-driven answers about loan approval patterns from a real-world loan dataset. Leveraging **semantic search** with vector databases and **local lightweight large language models (LLMs)**, this chatbot combines factual accuracy with natural conversational AI, all running entirely **locally and free of cost**.

---

## Features

- **Retrieve relevant information** from a loan dataset (CSV format) using semantic vector search (FAISS + Sentence Transformers).
- **Generate human-like responses** grounded on retrieved data using an open-source local LLM implemented via Hugging Face Transformers.
- **Interactive chat interface** built with Streamlit featuring modern, clean, and user-friendly design.
- **Source attribution:** Each answer shows which loan entries contributed to the response, enhancing transparency.
- **Fully open-source and zero API cost**, can run offline with modest hardware.

---

## Demo

Try the live chatbot here:  
👉 [Open the Chatbot Live on Streamlit](YOUR_LIVE_STREAMLIT_APP_LINK_HERE)

*Note: GPU-backed hosting is recommended for better performance.*

---

## Dataset

The chatbot uses the publicly available loan approval prediction dataset:  
[Loan Approval Dataset on Kaggle](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction?select=Training+Dataset.csv)

Ensure you download and place the dataset in the `data/` folder as `Training_Dataset.csv` before running the app locally.

---

## Getting Started

### Prerequisites

- Python 3.8+
- Recommended: A machine with at least 8GB RAM; GPU support enhances performance but is optional.

### Installation

1. Clone this repository:

   ```
   git clone https://github.com/yourusername/rag-loan-qa-chatbot.git
   cd rag-loan-qa-chatbot
   ```

2. (Optional) Create and activate a Python virtual environment:

   ```
   python3 -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```

3. Install required dependencies:

   ```
   pip install -r requirements.txt
   ```

4. Ensure your data is downloaded and placed:

   ```
   rag-loan-qa-chatbot/
   ├── data/
   │   └── Training_Dataset.csv  <-- place this file here
   ```

---

## Running the Chatbot Locally

Launch the Streamlit app with:

```
streamlit run app.py
```

This will spin up the chatbot interface accessible at `http://localhost:8501` in your browser.

---

## Project Structure


rag-loan-qa-chatbot/
├── app.py                      # Streamlit frontend UI
├── data/
│   └── Training_Dataset.csv    # Loan approval dataset CSV
├── requirements.txt            # Python dependencies
├── src/
│   ├── chatbot.py              # Core RAG pipeline: processing, retrieval, generation
│   ├── document_processor.py   # CSV to document processing utilities
│   ├── vector_store.py         # Vectorstore embedding & search helper
│   └── llm_handler.py          # Local LLM loading & text generation wrapper
└── README.md                   # This file


---

## How It Works

1. **Document Processing:** The CSV dataset is loaded and converted into text documents with structured metadata.
2. **Vector Embedding & Search:** Sentence-Transformers embed these documents into vector space and FAISS indexes them for similarity search.
3. **Question Answering (RAG):** Given a user query, the chatbot retrieves the top-k relevant documents, concatenates their content, and prompts a local LLM to generate a grounded answer.
4. **Result Display:** The generated response is shown along with the original document sources that informed the answer.

---

## Customization & Extensibility

- **Models:** Replace `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"` with any compatible Hugging Face causal LM for better or lighter inference.
- **Vector Store:** Swap FAISS with others like Chroma or Pinecone for scalable production.
- **Frontend:** Easily adapt `app.py` to Gradio or FastAPI backend.

---

## Troubleshooting & Tips

- Initial model downloads may take time — be patient on first run.
- On CPU-only machines, inference can be slow; a GPU speeds things up considerably.
- Make sure the dataset path in `app.py` matches your setup.

---

## Credits

- [LangChain](https://langchain.com)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Sentence-Transformers](https://www.sbert.net/)
- [FAISS](https://faiss.ai/)
- [Streamlit](https://streamlit.io/)
- Loan dataset courtesy Kaggle user Sonali Singh

---

## License

This project is released under the MIT License.
