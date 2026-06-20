import streamlit as st
import os

# set_page_config MUST be the first Streamlit command, otherwise the app
# errors on the initial load and only renders correctly after a refresh.
st.set_page_config(
    page_title="DocChat",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# --- Load the API key (after page config so the first load is clean) ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
    _key_ok = True
else:
    _key_ok = False

from rag_engine import process_document_to_chroma, answer_question


# --- Matte theme styling ---
st.markdown(
    """
    <style>
    /* Matte background gradient */
    .stApp {
        background: radial-gradient(1200px 800px at 20% -10%, #1d1d22 0%, #141416 55%, #101012 100%);
    }

    /* Main content width + breathing room */
    .block-container {
        max-width: 820px;
        padding-top: 3rem;
        padding-bottom: 7rem;
    }

    /* Hide default Streamlit chrome for a cleaner matte look */
    #MainMenu, footer, header [data-testid="stToolbar"] {
        visibility: hidden;
    }

    /* App title */
    .docchat-hero h1 {
        font-weight: 650;
        letter-spacing: -0.5px;
        margin-bottom: 0.15rem;
    }
    .docchat-hero p {
        color: #8f8b90;
        font-size: 0.96rem;
        margin-top: 0;
    }

    /* Sidebar matte surface */
    [data-testid="stSidebar"] {
        background-color: #131316;
        border-right: 1px solid #2a2a30;
    }
    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    /* Buttons: flat, matte, soft */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #3a3a42;
        background-color: #26262c;
        color: #e4e2e2;
        font-weight: 550;
        padding: 0.55rem 1rem;
        transition: background-color 0.15s ease, border-color 0.15s ease;
    }
    .stButton > button:hover {
        background-color: #30303a;
        border-color: #5a5470;
        color: #ffffff;
    }
    .stButton > button:active {
        background-color: #3a3548;
    }

    /* File uploader: matte dashed card */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #1b1b1f;
        border: 1px dashed #3a3a42;
        border-radius: 12px;
    }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        background-color: #1c1c21;
        border: 1px solid #2a2a30;
        border-radius: 14px;
        padding: 0.4rem 0.4rem;
        margin-bottom: 0.6rem;
    }

    /* Chat input: matte pill */
    [data-testid="stChatInput"] {
        background-color: #1b1b1f;
        border: 1px solid #2f2f37;
        border-radius: 14px;
    }
    [data-testid="stChatInput"] textarea {
        color: #e4e2e2;
    }

    /* Dividers softer */
    hr {
        border-color: #2a2a30 !important;
    }

    /* Empty-state placeholder card */
    .docchat-empty {
        text-align: center;
        color: #6f6b72;
        border: 1px dashed #2c2c33;
        border-radius: 16px;
        padding: 2.6rem 1.5rem;
        margin-top: 1.5rem;
        background-color: rgba(255,255,255,0.012);
    }
    .docchat-empty .icon { font-size: 2.2rem; }
    .docchat-empty .title { color: #b9b5bb; font-weight: 600; margin-top: 0.4rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- Header ---
st.markdown(
    """
    <div class="docchat-hero">
        <h1>🧠 DocChat</h1>
        <p>Upload a PDF and have a grounded conversation with its contents.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

if not _key_ok:
    st.error("Missing `GOOGLE_API_KEY` in `.streamlit/secrets.toml`")
    st.stop()


# --- Sidebar: document upload ---
with st.sidebar:
    st.markdown("### 📄 Your Document")
    st.caption("Upload a PDF, then click Process to index it.")
    uploaded_file = st.file_uploader("PDF file", type="pdf", label_visibility="collapsed")

    if st.button("Process"):
        if not uploaded_file:
            st.warning("Please upload a PDF file first.")
        else:
            with st.spinner("Reading and indexing your document..."):
                try:
                    message = process_document_to_chroma(uploaded_file)
                    st.session_state.doc_ready = True
                    st.success(message)
                except Exception as e:
                    st.error(f"Error processing document: {e}")

    st.divider()
    status = "✅ Document indexed" if st.session_state.get("doc_ready") else "⏳ No document yet"
    st.caption(status)


# --- Chat state ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Empty state placeholder
if not st.session_state.messages:
    st.markdown(
        """
        <div class="docchat-empty">
            <div class="icon">💬</div>
            <div class="title">Ask anything about your document</div>
            <div>Process a PDF from the sidebar, then type your question below.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Render history
for message in st.session_state.messages:
    avatar = "🧑" if message["role"] == "user" else "🧠"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# Handle input
if question := st.chat_input("Ask a question about your document..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("Thinking..."):
            try:
                response = answer_question(question)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {e}")
