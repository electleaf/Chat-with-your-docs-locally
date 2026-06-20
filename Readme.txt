<<<<<<< HEAD
# 🧠 DocChat: Chat With Your Documents

DocChat is an end-to-end **Retrieval-Augmented Generation (RAG)** application that lets you
upload a PDF and have a natural-language conversation with its contents. Ask questions and
get concise, context-grounded answers powered by Google Gemini.

Repository: https://github.com/electleaf/Chat-with-your-docs-locally


## ✨ Features

- **PDF ingestion** – upload any PDF and process it with one click.
- **Smart chunking** – documents are split into overlapping chunks (1000 chars / 200 overlap)
  for better retrieval accuracy.
- **Vector search** – embeddings are stored in a local **ChromaDB** vector store and queried
  by semantic similarity (top-5 chunks).
- **Conversational UI** – a clean Streamlit chat interface with persistent chat history via
  Session State.
- **Grounded answers** – the LLM answers strictly from retrieved context and says "I don't
  know" when the answer isn't present.


## 🛠️ Tech Stack

| Layer            | Technology                          |
| ---------------- | ----------------------------------- |
| UI               | Streamlit                           |
| Orchestration    | LangChain                           |
| LLM              | Google Gemini 2.0 Flash             |
| Embeddings       | Google `text-embedding-004`         |
| Vector store     | ChromaDB                            |
| PDF parsing      | PyPDF                               |


## 📂 Project Structure

```
Chat-with-your-docs-locally/
├── app.py            # Streamlit UI (upload, chat interface, session state)
├── rag_engine.py     # Core RAG logic: ingestion + retrieval/answer chain
├── check_models.py   # Helper script to list available Gemini models
├── requirements.txt  # Python dependencies
└── Readme.txt        # This file
```


## 🚀 Getting Started (Run Locally)

### 1. Clone the repo
```
git clone https://github.com/electleaf/Chat-with-your-docs-locally.git
cd Chat-with-your-docs-locally
```

### 2. Create a virtual environment & install dependencies
```
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux
pip install -r requirements.txt
```

### 3. Add your Google API key
Get a free key from https://aistudio.google.com/app/apikey, then create a file at
`.streamlit/secrets.toml`:

```
GOOGLE_API_KEY = "your_api_key_here"
```

### 4. Run the app
```
streamlit run app.py
```
The app opens at **http://localhost:8501**.


## 💡 How to Use

1. In the sidebar, upload a PDF and click **Process**.
2. Wait for the "Processed N chunks successfully!" confirmation.
3. Type a question in the chat box and get an answer grounded in your document.


## 🌐 Deploy a Live (Shareable) Link

To get a public link anyone can open, deploy free on **Streamlit Community Cloud**:

1. Push this repo to GitHub (already done at the link above).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **New app**, select this repo, branch `main`, and set the main file to `app.py`.
4. Under **Advanced settings → Secrets**, add:
   ```
   GOOGLE_API_KEY = "your_api_key_here"
   ```
5. Click **Deploy**. Your live app URL will look like:
   `https://<your-app-name>.streamlit.app`


## 📝 Notes

- The `.streamlit/secrets.toml` file is git-ignored to keep your API key private.
- The local vector store is persisted to `./chroma_db`.
=======
# DocuChat: Enterprise-Grade Private RAG Knowledge Base

![Python](https://img.shields.io/badge/Python-3.11-blue) ![Streamlit](https://img.shields.io/badge/UI-Streamlit-red) ![LangChain](https://img.shields.io/badge/Orchestration-LangChain-green) ![Gemini](https://img.shields.io/badge/AI-Gemini%202.0%20Flash-purple) ![ChromaDB](https://img.shields.io/badge/Database-ChromaDB-orange)

DocuChat is a privacy-first Retrieval-Augmented Generation (RAG) application. It enables users to query complex PDF documents using natural language, leveraging the low-latency Google Gemini 2.0 Flash model for rapid inference.

## Problem Statement & Business Value
Standard Large Language Models (LLMs) are prone to hallucination when queried about proprietary or private datasets. DocuChat mitigates this risk by grounding the generation process exclusively in user-provided documentation. This ensures high-fidelity, context-aware responses while maintaining data privacy, as document vectors are stored entirely on local infrastructure.

## Architecture Flow
This application utilizes a robust LangChain orchestration pipeline:

1. Ingestion: Raw PDF documents are parsed using `PyPDFLoader` and tokenized into overlapping 1000-character chunks via `RecursiveCharacterTextSplitter` to optimize for LLM context window constraints.
2. Embedding & Storage: Text chunks are transformed into high-dimensional dense vectors using Google's `text-embedding-004` model and persisted to disk using ChromaDB for local, low-latency vector search.
3. Retrieval: Upon receiving a user query, the system performs a similarity search ($k=5$) against the vector database to retrieve the most mathematically relevant document fragments.
4. Generation: The retrieved context and the initial query are injected into a strict prompt template and processed by `gemini-2.0-flash` to synthesize a factual, hallucination-free response.

## Tech Stack
* Frontend: Streamlit
* AI Orchestration: LangChain Core & Community
* Vector Database: ChromaDB 
* LLM & Embeddings: Google Generative AI (`gemini-2.0-flash`, `text-embedding-004`)
>>>>>>> 69ccc29e0db2902c21c24aa34fa1feb57cd31a11
