"""
vector_store.py
----------------
This file is responsible for:
1. Taking the text chunks created from the PDF.
2. Converting those chunks into embeddings (numeric vectors).
3. Storing those vectors inside a FAISS vector database.
4. Allowing future loading + similarity search.

Originally, I tried Chroma, but Windows could not load the
chromadb_rust_bindings DLL → constant import errors.
So I switched to FAISS, which is stable and works offline.
"""

from typing import List
import os

from langchain_core.documents import Document

# FAISS = Local vector store (no DLL issues on Windows, reliable)
from langchain_community.vectorstores import FAISS

# Embeddings provider → Using Gemini API (text-embedding-004)
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Read environment variables from .env file
from dotenv import load_dotenv
load_dotenv()


# ------------------ CONFIGURATION ------------------

# Where FAISS index will be saved
FAISS_DIR = "data/faiss_index"

# Gemini embedding model (very good quality)
EMBEDDING_MODEL = "models/text-embedding-004"


# ------------------ EMBEDDING MODEL LOADER ------------------

def get_embedding_model():
    """
    Create and return the embedding model.
    This model converts text chunks → numerical vectors.

    NOTE:
    - Requires GOOGLE_API_KEY in .env
    - No GPU or PyTorch required (Gemini API runs in cloud)
    """

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Add it to your .env file.")

    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key
    )


# ------------------ BUILD FAISS VECTOR STORE ------------------

def build_vector_store(chunks: List[Document], persist_directory: str = FAISS_DIR):
    """
    Takes the list of Document chunks and builds a FAISS vector store.

    Steps:
    1. Convert text chunks → embeddings using Gemini.
    2. Store them inside FAISS (in-memory database).
    3. Save the FAISS index to disk so the user does NOT need to rebuild every time.
    """

    print("Creating embedding model...")
    embeddings = get_embedding_model()

    print("Building FAISS index (this may take time on first run)...")
    vectordb = FAISS.from_documents(chunks, embeddings)

    # Ensure folder exists
    os.makedirs(persist_directory, exist_ok=True)

    # Save FAISS DB to disk
    vectordb.save_local(persist_directory)

    return vectordb


# ------------------ LOAD EXISTING FAISS STORE ------------------

def load_vector_store(persist_directory: str = FAISS_DIR):
    """
    Load a previously saved FAISS index from disk.
    This avoids recomputing embeddings every time.
    """

    embeddings = get_embedding_model()
    vectordb = FAISS.load_local(
        persist_directory,
        embeddings,
        allow_dangerous_deserialization=True  # required for FAISS loading
    )

    return vectordb


# ------------------ TEST BLOCK (runs only when executed directly) ------------------

if __name__ == "__main__":
    """
    Manual test:
    1. Load the PDF
    2. Split into chunks
    3. Build FAISS vector DB
    4. Run a sample similarity search query
    """

    from pdf_loader import load_pdf_pages
    from chunker import make_chunks

    print("Loading pages...")
    pages = load_pdf_pages("data/Arpit_Negi_Resume.pdf")
    print("Pages:", len(pages))

    print("Chunking...")
    chunks = make_chunks(pages)
    print("Chunks:", len(chunks))

    print("Building FAISS vector store...")
    vectordb = build_vector_store(chunks)
    print("Vector store saved!")

    # Simple search test
    query = "ML Developer having 3 year experience in Python and Data Science?"
    results = vectordb.similarity_search(query, k=3)

    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(doc.metadata)
        print(doc.page_content[:300], "\n")
