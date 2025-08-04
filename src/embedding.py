from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def create_vector_store(documents, dir = "./data/vector_store"):
    if not os.path.exists(dir):
        os.mkdir(dir)

    test_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 512,
        chunk_overlap = 50,
        add_start_index = True
    )

    chunks = test_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

    vector_store = FAISS.from_documents(chunks,embeddings)

    return vector_store
    
