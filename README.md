```markdown


  

💎 RAG Q&A Chatbot for Loan Approval Prediction

  An intelligent chatbot for exploring real loan application data using Retrieval-Augmented Generation.


---

## 🚀 Live Demo

👉 Open the Chatbot Live on Streamlit

*GPU hosting recommended for smoothest answers!*

---

## 🧠 What is this Project?

This repo contains a modern **Retrieval-Augmented Generation (RAG) Q&A Chatbot** trained on a real loan approval dataset. It helps you query patterns, statistics, and sample cases around loan applications and approvals, answering with clear explanations—**all powered by local, open-source AI with zero API cost**.

---

## 🎯 Features

- 🔎 **Semantic Search:** Find the most relevant records in your dataset with FAISS & Sentence Transformers.
- 🤖 **Grounded AI Answers:** Local LLM (e.g., TinyLlama) writes answers based on the actual retrieved data, not just guesses.
- 💬 **Sleek Chat Interface:** Beautiful Streamlit frontend with modern UX, avatars, and transparent sources.
- 🏷 **Source Highlighting:** See which data rows the answer used—ensuring traceability.
- 🛡 **100% Free & Local:** No API keys, no cloud dependencies, runs anywhere!

---

## 📊 Dataset

Curated from Kaggle:  
[Loan Approval Prediction Dataset](https://www.kaggle.com/datasets/sonalisingh1411/loan-approval-prediction?select=Training+Dataset.csv)  
Just place `Training_Dataset.csv` in the `data/` folder before use.

---

## 💾 Quickstart

**1. Clone & Prepare**

```
git clone https://github.com/yourusername/rag-loan-qa-chatbot.git
cd rag-loan-qa-chatbot
```

**2. (Optional) Activate a Virtual Environment**

```
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
```

**3. Install Dependencies**

```
pip install -r requirements.txt
```

**4. Add Dataset**

Download the CSV and place as:
```
/data/Training_Dataset.csv
```

**5. Run the App**

```
streamlit run app.py
```
Browse to `http://localhost:8501` to chat!

---

## 🗂 Project Structure
```
rag-loan-qa-chatbot/
│
├── app.py                  # Streamlit UI (main entry)
├── requirements.txt        # All dependencies
├── data/
│   └── Training_Dataset.csv
└── src/
    ├── chatbot.py              # Full RAG pipeline (processing, retrieval, generation)
    ├── document_processor.py   # Document formatting utility
    ├── vector_store.py         # Vector search logic (FAISS + embeddings)
    └── llm_handler.py          # Local LLM helper (Hugging Face/Transformers)
```

---

## 🛠️ How It Works

1. **Ingest & Process:** CSV data → semantic documents with metadata.
2. **Embed & Index:** Documents embedded, stored in FAISS for fast similarity search.
3. **Retrieve & Generate:** For each user question:  
   - Find top-matching entries  
   - Concatenate info as context  
   - LLM generates a clear, relevant answer using only those facts.
4. **Display & Explain:** User sees the answer **and** which data rows the answer came from.

---

## ✨ Example Questions

- What factors most influence loan approval?
- Give examples of high-income rejected applications.
- How does property area affect approval rates?
- List characteristics of approved loans for married applicants.

---

## ⚙️ Customization

- **LLM:** Swap `"TinyLlama/TinyLlama-1.1B-Chat-v1.0"` for any Hugging Face causal LM you prefer.
- **Vector DB:** Use Chroma or Pinecone for larger/production setups instead of FAISS.
- **UI:** Adapt `app.py` for Gradio or other frameworks if you wish.

---

## 🙏 Credits

- [LangChain](https://langchain.com)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Sentence-Transformers](https://www.sbert.net/)
- [FAISS](https://faiss.ai/)
- [Streamlit](https://streamlit.io/)
- Kaggle dataset by Sonali Singh

---

## 📝 License

MIT License

---


  Built with 🩵 for the open-source AI community!
  Questions or improvements? Create an issue or PR any time.

```
Replace `YOUR_LIVE_STREAMLIT_APP_LINK_HERE` with your live app’s link before publishing!
