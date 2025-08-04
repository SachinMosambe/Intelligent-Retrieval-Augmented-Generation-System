from dotenv import load_dotenv
import os

load_dotenv()

token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

print("=== TEST SCRIPT STARTED ===")
print("Loaded Token:", token)

# Add any other code you want to test below
from langchain.embeddings import HuggingFaceEmbeddings

try:
    embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    print("Embedding model loaded successfully.")
except Exception as e:
    print("Error loading embedding model:", e)

print("=== TEST SCRIPT COMPLETED ===")
