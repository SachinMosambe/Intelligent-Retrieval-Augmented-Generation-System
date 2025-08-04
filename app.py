import streamlit as st
from src.loader import load_files, load_urls
from src.embedding import create_vector_store
from src.retriever import load_vectorstore, search_vector_store
from src.generator import generate_response


st.set_page_config(page_title = "Multi RAG chatbot", layout= "wide")
st.title("Ask question from you dogs or url")

st.sidebar.header("Upload your files or provide URLs")
option = st.sidebar.radio("Select an option:",("upload files", "Enter URLs"))

documents = []

if option == "upload files":
    uploaded_files = st.sidebar.file_uploader("Choose_files", type = ["pdf", "docx", "txt"], accept_multiple_files = True)

    if uploaded_files:
        file_paths = []
        for files in uploaded_files:
            file_path = f"./data/{files.name}"
            with open(file_path,"wb") as f:
                f.write(files.getbuffer())
            file_paths.append(file_path)
        
        documents = load_files(file_paths)
        st.success("Files loaded successfully!")
    
elif option == "Enter URLs":
    urls = st.sidebar.text_area("Enter urls (sepeated by comma)")

    if urls:
        url_list = [urls.strip() for url in urls.split(',')]
        documents = load_urls(url_list)
        st.success("URLs loaded successfully!")

if documents:
    with st.spinner("Creating vector store..."):
        vector_store = create_vector_store(documents)   
        retriever = search_vector_store(vector_store)
    st.success("Vector store created successfully!")


    query = st.text_input("Ask a question related to upload files or URLs")

    if query:
        with st.spinner("Generating response..."):
            response = generate_response(retriever,query)
        st.markdown(f"Answer: {response}")


else:
    st.info("Please upload files or enter URLs to proceed.")
           
