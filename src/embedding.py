from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def create_vector_store_enhanced(documents, dir="./data/vector_store"):
    """
    Enhanced vector store with optimized chunking and metadata
    """
    if not os.path.exists(dir):
        os.makedirs(dir)
    
    # Optimized chunking strategy
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
        add_start_index=True
    )
    
    # Split with metadata
    chunks = []
    for doc in documents:
        splits = text_splitter.split_documents([doc])
        
        for i, chunk in enumerate(splits):
            chunk.metadata.update({
                "chunk_id": i,
                "source": doc.metadata.get("source", "unknown"),
                "total_chunks": len(splits)
            })
        chunks.extend(splits)
    
    print(f"📄 Created {len(chunks)} chunks from {len(documents)} documents")
    
    # Use optimized embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # Create FAISS vector store
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    return vector_store
