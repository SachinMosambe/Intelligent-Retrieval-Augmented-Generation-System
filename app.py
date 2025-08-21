import streamlit as st
from src.loader import load_files, load_urls
from src.embedding import create_vector_store
from src.retriever import search_vector_store
from src.generator import generate_response

st.set_page_config(page_title="Multi RAG chatbot", layout="wide")
st.title("Ask question from your dogs or URLs")

st.sidebar.header("Upload your files or provide URLs")
option = st.sidebar.radio("Select an option:", ("Upload files", "Enter URLs"))

# Initialize session state variables for documents, vector store, retriever, and chat
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'retriever' not in st.session_state:
    st.session_state.retriever = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Reset query input and chat history when switching options
if 'previous_option' not in st.session_state:
    st.session_state.previous_option = option

if st.session_state.previous_option != option:
    st.session_state.query = ""
    st.session_state.chat_history = []
    st.session_state.previous_option = option

# Handle file upload or URL input
if option == "Upload files":
    uploaded_files = st.sidebar.file_uploader("Choose files", type=["pdf", "docx", "txt"], accept_multiple_files=True)

    if uploaded_files:
        file_paths = []
        for file in uploaded_files:
            file_path = f"./data/{file.name}"
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            file_paths.append(file_path)
        
        # Load files only if they haven't been loaded already
        if not st.session_state.documents:
            st.session_state.documents = load_files(file_paths)
            st.success("Files loaded successfully!")
        
elif option == "Enter URLs":
    urls = st.sidebar.text_area("Enter URLs (separated by commas)")

    if urls:
        url_list = [url.strip() for url in urls.split(',') if url.strip()]  # Remove any empty URLs
        if not url_list:
            st.error("Please enter valid URLs.")
        else:
            st.write("Processing the following URLs:")
            st.write(url_list)  # Debugging output

            # Load URLs only if they haven't been loaded already
            if not st.session_state.documents:
                try:
                    st.session_state.documents = load_urls(url_list)
                    st.success(f"URLs loaded successfully! Found {len(st.session_state.documents)} documents.")
                except Exception as e:
                    st.error(f"Error loading URLs: {str(e)}")

# Create vector store and retriever only if they haven't been created already
if st.session_state.documents and not st.session_state.vector_store:
    with st.spinner("Creating vector store..."):
        st.session_state.vector_store = create_vector_store(st.session_state.documents)
        st.session_state.retriever = search_vector_store(st.session_state.vector_store)
    st.success("Vector store created successfully!")

# Clear chat button
if st.button("Clear Chat"):
    st.session_state.chat_history = []
    st.session_state.query = ""
    st.rerun()  # Reset the app state and rerun the app

# Display chat history (if any)
if st.session_state.chat_history:
    for entry in st.session_state.chat_history:
        st.write(f"**User:** {entry['question']}")
        st.write(f"**Answer:** {entry['answer']}")

# Query input
query = st.text_input("Ask a question related to uploaded files or URLs", key="query")

if query:
    if st.session_state.retriever:
        with st.spinner("Generating response..."):
            response = generate_response(st.session_state.retriever, query)
        st.session_state.chat_history.append({"question": query, "answer": response})
        st.markdown(f"**Answer:** {response}")
    else:
        st.error("Vector store or retriever is not available. Please load documents first.")
else:
    st.info("Please upload files or enter URLs to proceed.")
