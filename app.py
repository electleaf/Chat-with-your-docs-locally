import streamlit as st
import os

# Set the API key environment variable so LangChain can find it
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
else:
    st.error("Missing GOOGLE_API_KEY in .streamlit/secrets.toml")
from rag_engine import process_document_to_chroma, answer_question

# Page title and icon
st.set_page_config(page_title="DocChat", page_icon="🧠")
# Title of the app
st.title("🧠 DocChat: Chat With Your Documents")



# --- 1. The File Uploader ---
# Put this in a sidebar to keep the main chat area clean
with st.sidebar:
    st.header("Your Document")
    uploaded_file = st.file_uploader("Upload your PDF and click 'Process'", type="pdf")

    if st.button("Process"):
        if not uploaded_file:
            st.error("Please upload a PDF file")
            st.stop()
        else:
            with st.spinner("Processing your document..."):
                try:
                    message = process_document_to_chroma(uploaded_file)
                    st.success(message)
                except Exception as e:
                    st.error(f"Error processing document: {e}")

# --- 2. The Chat Interface ---
st.divider()

# Initialize Chat History (So messages stay when you click things)
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
if question := st.chat_input("Ask a question about your document..."):
    # 1. Display User Message
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})

    # 2. Generate & Display AI Response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # CALL THE NEW FUNCTION
                response = answer_question(question)
                st.markdown(response)
                # Save AI response to history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"An error occurred: {e}")