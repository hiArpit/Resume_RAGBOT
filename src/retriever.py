from langchain_community.vectorstores import FAISS
# Importing FAISS vectorstore class from LangChain community package
# FAISS is an efficient on-disk/in-memory nearest-neighbour search index used for similarity search of embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# Calling a class that lets you call Gemini embeddings via langchain-compatible interface
import os
from dotenv import load_dotenv
load_dotenv()

def load_retriever(db_path="data/faiss_index"):
    # This function will load faiss index from disk and return retriever object you can use
    # to get nearest chunks for a query.
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        # To check whether key is present or not
        raise ValueError("Google_API_KEY not set in .env")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=api_key
    )

    vectordb = FAISS.load_local(
        folder_path=db_path,
        # loads the previously saved FAISS index from folder_path
        embeddings=embeddings,
        # attach your embeddings wrapper so the vectordb can embed queries in the same way like the text
        allow_dangerous_deserialization=True
        # Flag that allows loading serialized Python objects saved into the index
    )
    retriever = vectordb.as_retriever(search_kwargs={"k": 5})
    # Converting the FAISS vectorstore into a retriever object
    # k=5 makes sure that the top 5 closest chunks be returned for any query

    return retriever