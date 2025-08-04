from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def load_vectorstore(dir = "./data/vector_store"):
    embedder = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.load_local(dir, embedder)
    return vectorstore

def search_vector_store(vectorstore):
    retriever = vectorstore.as_retriever(search_type = "similarity", search_kwargs = {'k': 5})
    return retriever
    