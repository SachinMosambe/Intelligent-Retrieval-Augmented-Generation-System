from langchain_community.document_loaders import TextLoader, PyPDFLoader, UnstructuredWordDocumentLoader, SeleniumURLLoader
import os
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_files(file_paths):
    documents = []
    for file_path in file_paths:
        try:
            if file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
            elif file_path.endswith('.pdf'):
                loader = PyPDFLoader(file_path)
                documents.extend(loader.load())
            elif file_path.endswith('.docx'):
                loader = UnstructuredWordDocumentLoader(file_path)
                documents.extend(loader.load())
            else:
                logger.warning(f"Unsupported file format for: {file_path}")
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            continue
    return documents

def load_urls(urls):
    documents = []
    try:
        # Ensure urls are not empty
        if not urls:
            logger.error("No URLs provided.")
            return documents
        
        # Use the Selenium URL loader to fetch the content from the URLs
        loader = SeleniumURLLoader(urls=urls)
        documents = loader.load()
        
        # Check if documents are returned
        if not documents:
            logger.warning("No documents found for the provided URLs.")
            
    except Exception as e:
        logger.error(f"Error loading URLs: {e}")
    
    return documents
