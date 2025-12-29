import streamlit as st
from src.loader import load_files, load_urls
from src.embedding import create_vector_store_enhanced
from src.retriever import search_vector_store_enhanced
from src.generator import generate_response_enhanced
from src.explainability import RAGExplainer
import os
import shutil

# Page config
st.set_page_config(
    page_title="Advanced RAG System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; color: #1E88E5; font-weight: bold;}
    .sub-header {font-size: 1.2rem; color: #666;}
    .source-box {background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin: 10px 0;}
    .answer-box {background-color: #e8f5e9; padding: 15px; border-radius: 5px; margin: 10px 0;}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🤖 Advanced RAG System</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multi-Source Document Q&A with Explainability</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("📁 Data Source Configuration")
option = st.sidebar.radio("Select Input Method:", ("📄 Upload Files", "🌐 Enter URLs"))

# Advanced settings
with st.sidebar.expander("⚙️ Advanced Settings"):
    enable_stopwords = st.checkbox("Remove Stopwords", value=False)
    enable_explainability = st.checkbox("Show Explainability", value=True)
    top_k_docs = st.slider("Top K Documents", 3, 10, 5)

# Initialize session state
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processed' not in st.session_state:
    st.session_state.processed = False

# File upload
if option == "📄 Upload Files":
    uploaded_files = st.sidebar.file_uploader(
        "Upload Documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files and not st.session_state.processed:
        os.makedirs("./data/uploads", exist_ok=True)
        file_paths = []
        
        for file in uploaded_files:
            file_path = f"./data/uploads/{file.name}"
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            file_paths.append(file_path)
        
        with st.spinner("📖 Loading and preprocessing documents..."):
            st.session_state.documents = load_files(file_paths, preprocess=True)
        
        if st.session_state.documents:
            st.sidebar.success(f"✅ Loaded {len(st.session_state.documents)} documents")

# URL input
elif option == "🌐 Enter URLs":
    urls_input = st.sidebar.text_area("Enter URLs (one per line)", height=150)
    
    if urls_input and not st.session_state.processed:
        url_list = [url.strip() for url in urls_input.split('\n') if url.strip()]
        
        with st.spinner("🌐 Fetching content from URLs..."):
            st.session_state.documents = load_urls(url_list, preprocess=True)
        
        if st.session_state.documents:
            st.sidebar.success(f"✅ Loaded {len(st.session_state.documents)} documents")

# Create vector store
if st.session_state.documents and not st.session_state.vector_store:
    with st.spinner("🔧 Building vector store with embeddings..."):
        st.session_state.vector_store = create_vector_store_enhanced(st.session_state.documents)
        st.session_state.retriever = search_vector_store_enhanced(st.session_state.vector_store)
        st.session_state.processed = True
    
    st.sidebar.success("✅ System ready!")
    st.sidebar.info(f"📊 Indexed {len(st.session_state.documents)} documents")

# Main interface
st.markdown("---")

col1, col2 = st.columns([5, 1])

with col1:
    query = st.text_input("💬 Ask a question:", placeholder="What would you like to know?")

with col2:
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()

# Process query
if query and st.session_state.retriever:
    with st.spinner("🤔 Thinking..."):
        result = generate_response_enhanced(st.session_state.retriever, query)
    
    # Display answer
    st.markdown('<div class="answer-box">', unsafe_allow_html=True)
    st.markdown(f"**Answer:** {result['answer']}")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display sources
    if result['sources']:
        st.markdown('<div class="source-box">', unsafe_allow_html=True)
        st.markdown(f"**📚 Sources ({result['num_sources']}):**")
        for i, source in enumerate(result['sources'], 1):
            st.markdown(f"{i}. {source}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Explainability (if enabled)
    if enable_explainability:
        with st.expander("🔍 View Retrieval Explainability"):
            docs = st.session_state.retriever.retrieve_and_rerank(query, k=10, top_n=top_k_docs)
            
            # Show document previews
            st.markdown("**Retrieved Documents:**")
            for i, doc in enumerate(docs, 1):
                st.text_area(
                    f"Document {i}",
                    doc.page_content[:300] + "...",
                    height=100,
                    key=f"doc_{i}"
                )
    
    # Add to chat history
    st.session_state.chat_history.append({
        "question": query,
        "answer": result['answer'],
        "sources": result['sources']
    })

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("### 📜 Chat History")
    
    for i, entry in enumerate(reversed(st.session_state.chat_history[-5:]), 1):
        with st.expander(f"Q{i}: {entry['question'][:50]}..."):
            st.markdown(f"**Q:** {entry['question']}")
            st.markdown(f"**A:** {entry['answer']}")
            if entry['sources']:
                st.markdown(f"**Sources:** {', '.join(entry['sources'])}")

# Info box
if not st.session_state.processed:
    st.info("👈 Upload documents or enter URLs to get started!")