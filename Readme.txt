DocChat: Chat With Your Documents
=================================

DocChat is an end-to-end Retrieval-Augmented Generation (RAG) application that lets you
upload a PDF and have a natural-language conversation with its contents. Ask questions and
get concise, context-grounded answers powered by Google Gemini.

Repository: https://github.com/electleaf/Chat-with-your-docs-locally


Features
--------
- PDF ingestion: upload any PDF and index it with one click.
- Smart chunking: documents are split into overlapping chunks (1000 chars / 200 overlap)
  for better retrieval accuracy.
- Vector search: embeddings are stored in a local ChromaDB vector store and queried by
  semantic similarity (top-5 chunks).
- Conversational UI: a clean, dark "matte" Streamlit chat interface with persistent chat
  history via Session State.
- Grounded answers: the LLM answers strictly from retrieved context and says it does not
  know when the answer is not present.


Tech Stack
----------
- UI:            Streamlit
- Orchestration: LangChain
- LLM:           Google Gemini 2.5 Flash
- Embeddings:    Google gemini-embedding-001
- Vector store:  ChromaDB
- PDF parsing:   PyPDF


Project Structure
-----------------
Chat-with-your-docs-locally/
    app.py                  Streamlit UI (theme, upload, chat interface, session state)
    rag_engine.py           Core RAG logic: ingestion + retrieval/answer chain
    check_models.py         Helper script to list available Gemini models
    requirements.txt        Python dependencies
    Readme.txt              This file
    PROJECT_FLOW.txt        Detailed end-to-end explanation of how the project works
    .streamlit/
        config.toml         Streamlit theme (dark matte) and server settings
        secrets.toml        Your GOOGLE_API_KEY (git-ignored, you create this)


Getting Started (Run Locally)
-----------------------------
1. Clone the repo:
       git clone https://github.com/electleaf/Chat-with-your-docs-locally.git
       cd Chat-with-your-docs-locally

2. Create a virtual environment and install dependencies:
       python -m venv venv
       venv\Scripts\activate            (Windows)
       source venv/bin/activate         (macOS / Linux)
       pip install -r requirements.txt

3. Add your Google API key. Get a free key from
   https://aistudio.google.com/app/apikey, then create a file at
   .streamlit/secrets.toml containing:
       GOOGLE_API_KEY = "your_api_key_here"

4. Run the app:
       streamlit run app.py

   The app opens at http://localhost:8501


How to Use
----------
1. In the sidebar, upload a PDF and click "Process".
2. Wait for the "Processed N chunks successfully!" confirmation.
3. Type a question in the chat box and get an answer grounded in your document.


Deploy a Live (Shareable) Link
------------------------------
To get a public link anyone can open, deploy free on Streamlit Community Cloud:
1. Push this repo to GitHub (already done at the link above).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "New app", select this repo, branch "main", main file "app.py".
4. Under Advanced settings > Secrets, add:
       GOOGLE_API_KEY = "your_api_key_here"
5. Click Deploy. Your live app URL will look like:
       https://<your-app-name>.streamlit.app


Recent Changes / Fixes
----------------------
- Embedding model updated from "models/text-embedding-004" to "models/gemini-embedding-001".
  The old model now returns a 404 and broke PDF ingestion.
- Chat model updated from "gemini-2.0-flash" to "gemini-2.5-flash". The 2.0 model was
  removed from the free tier and returned 429 quota errors when answering.
- Fixed a double-refresh bug on startup: st.set_page_config() is now the first Streamlit
  command in app.py, so the app renders correctly on the first load.
- Added a dark "matte" UI theme via .streamlit/config.toml plus custom CSS in app.py
  (flat buttons, matte chat bubbles, custom avatars, sidebar status, and an empty state).
- Added requirements.txt so the project can be installed and deployed reliably.


Notes
-----
- The .streamlit/secrets.toml file is git-ignored to keep your API key private.
- The local vector store is persisted to the ./chroma_db folder.
- Free-tier Gemini quotas are limited. If you upload a large PDF or ask several questions
  in quick succession, you may briefly see a 429 quota error; wait 30-60 seconds and retry.
