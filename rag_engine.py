import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION ---
# We use the same embedding model as before
EMBEDDING_MODEL = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# 1. INGESTION FUNCTION (You already built this)
def process_document_to_chroma(uploaded_file):
    """
    Takes a PDF, splits it, and stores it in ChromaDB.
    """
    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_file_path = temp_file.name

    # Load and split
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    os.remove(temp_file_path)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    splits = text_splitter.split_documents(docs)
    
    # Store in ChromaDB
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=EMBEDDING_MODEL,
        persist_directory="./chroma_db"
    )
    
    return f"✅ Processed {len(splits)} chunks successfully!"

# 2. RETRIEVAL FUNCTION (This is NEW!)
def answer_question(question):
    """
    Takes a user question, finds relevant context, and generates an answer.
    """
    # A. Connect to the existing Database
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=EMBEDDING_MODEL)
    
    # B. Create a Retriever (It will look for the top 5 most similar chunks)
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    
    # C. Initialize the LLM (Gemini 2.0 Flash - Fast & Free)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
    
    # D. Create the Prompt Template
    system_prompt = (
        "You are a helpful assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    # E. Build the Chain
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    # F. Run the Chain
    response = rag_chain.invoke({"input": question})
    
    return response["answer"]