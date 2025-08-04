from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader,PlaywrightURLLoader
import os

def load_files(file_paths):
    documents = []

    for file_path in file_paths:
        if file_path.endswith('.txt'):
            loader = TextLoader(file_path, encoding ='utf-8')
            documents.extend(loader.load())
        elif file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file_path.endswith('.docx'):
            loader = UnstructuredWordDocumentLoader(file_path)
            documents.extend(loader.load())

    return documents


def load_urls(urls):
    loader = PlaywrightURLLoader(urls = urls, remove_selectors=["header", "footer", "nav", ".ads"])
    return loader.load()

