from langchain_community.document_loaders import TextLoader, PyPDFLoader, Docx2txtLoader
import os
import logging
import re
from pathlib import Path
import pytesseract
from PIL import Image
from pdf2image import convert_from_path
from langchain.schema import Document as LangchainDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# STOPWORDS for text cleaning
STOPWORDS = set([
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he',
    'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'will', 'with'
])

def clean_text_with_regex(text):
    """
    Enhanced text cleaning with regex patterns
    """
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-]', '', text)
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    return text.strip()

def remove_stopwords(text):
    """
    Remove common stopwords from text
    """
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in STOPWORDS]
    return ' '.join(filtered_words)

def preprocess_document_text(text, remove_stops=False):
    """
    Complete preprocessing pipeline
    """
    # Clean with regex
    text = clean_text_with_regex(text)
    # Optionally remove stopwords
    if remove_stops:
        text = remove_stopwords(text)
    return text

def extract_text_from_image(image_path):
    """
    Extract text from image using Tesseract OCR (lightweight)
    """
    try:
        img = Image.open(image_path)
        # Use Tesseract for OCR
        text = pytesseract.image_to_string(img, lang='eng')
        return text
    except Exception as e:
        logger.error(f"OCR failed for {image_path}: {e}")
        return ""

def extract_text_from_scanned_pdf(pdf_path):
    """
    Extract text from scanned PDF using Tesseract OCR (lightweight)
    """
    try:
        # Convert PDF to images (DPI=200 for balance between quality and memory)
        images = convert_from_path(pdf_path, dpi=200)
        
        extracted_text = []
        for i, img in enumerate(images):
            logger.info(f"OCR processing page {i+1}/{len(images)} of {pdf_path}")
            text = pytesseract.image_to_string(img, lang='eng')
            extracted_text.append(text)
        
        return "\n\n".join(extracted_text)
    except Exception as e:
        logger.error(f"Scanned PDF OCR failed for {pdf_path}: {e}")
        return ""

def load_files(file_paths, preprocess=True):
    """
    Load files optimized for AWS Free Tier (low memory usage)
    Supports: .txt, .pdf (text & scanned), .doc, .docx, .png, .jpg, .jpeg, .tiff
    """
    documents = []
    
    # Image extensions for OCR
    image_extensions = {'.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif'}
    
    for file_path in file_paths:
        try:
            file_ext = Path(file_path).suffix.lower()
            
            if file_path.endswith('.txt'):
                loader = TextLoader(file_path, encoding='utf-8')
                docs = loader.load()
                
            elif file_path.endswith('.pdf'):
                # Try regular PDF extraction first
                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    
                    # Check if PDF is scanned (empty or very little text)
                    total_text = ''.join([doc.page_content for doc in docs])
                    
                    if len(total_text.strip()) < 50:  # Likely scanned PDF
                        logger.info(f"Detected scanned PDF, using lightweight OCR: {file_path}")
                        ocr_text = extract_text_from_scanned_pdf(file_path)
                        docs = [LangchainDocument(
                            page_content=ocr_text,
                            metadata={"source": file_path, "method": "ocr"}
                        )]
                        
                except Exception as e:
                    logger.warning(f"Regular PDF extraction failed, trying OCR: {e}")
                    ocr_text = extract_text_from_scanned_pdf(file_path)
                    docs = [LangchainDocument(
                        page_content=ocr_text,
                        metadata={"source": file_path, "method": "ocr"}
                    )]
                    
            elif file_path.endswith('.docx') or file_path.endswith('.doc'):
                # Docx2txtLoader is lightweight and works for both formats
                loader = Docx2txtLoader(file_path)
                docs = loader.load()
                
            elif file_ext in image_extensions:
                # Handle scanned images with lightweight OCR
                logger.info(f"Processing image with OCR: {file_path}")
                ocr_text = extract_text_from_image(file_path)
                docs = [LangchainDocument(
                    page_content=ocr_text,
                    metadata={"source": file_path, "method": "ocr"}
                )]
                
            else:
                logger.warning(f"Unsupported file format: {file_path}")
                continue
            
            # Preprocess if enabled
            if preprocess:
                for doc in docs:
                    if doc.page_content:
                        doc.page_content = preprocess_document_text(doc.page_content)
            
            documents.extend(docs)
            logger.info(f"✅ Loaded: {file_path} ({len(docs)} document(s))")
            
        except Exception as e:
            logger.error(f"❌ Error loading {file_path}: {e}")
            continue
    
    return documents