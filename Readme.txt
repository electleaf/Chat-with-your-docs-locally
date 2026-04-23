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
