# IntelliRAG 🧠

**Intelligent Retrieval-Augmented Generation System**

A production-ready RAG system with advanced retrieval, cross-encoder reranking, and comprehensive RAGAS evaluation metrics.

---

## ✨ Features

- 🔍 **Advanced Retrieval** - FAISS vector store with semantic search
- 🎯 **Cross-Encoder Reranking** - MS-MARCO model for improved relevance
- 📊 **Query Expansion** - Automatic query variations for better recall
- 🤖 **LLM Integration** - Meta-Llama 3.1 via Together AI
- 📈 **RAGAS Evaluation** - Comprehensive quality metrics
- 📄 **Multi-Format Support** - PDF, DOCX, TXT, images with OCR
- 🔬 **Explainability** - Retrieval visualization and source attribution

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/intellirag.git
cd intellirag

# Install dependencies
pip install -r requirements.txt

# Set up environment
echo "TOEGETHERAI_API_KEY=your_api_key_here" > .env
```

### Basic Usage

```python
from src.embedding import create_vector_store_enhanced
from src.generator import generate_response_enhanced
from src.loader import load_files
from src.retriever import search_vector_store_enhanced

# Load documents
documents = load_files(["./data/your_document.pdf"])

# Create vector store
vector_store = create_vector_store_enhanced(documents)

# Create retriever
retriever = search_vector_store_enhanced(vector_store)

# Ask question
result = generate_response_enhanced(retriever, "What is the main topic?")

print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

---

## 📊 Evaluation

Run comprehensive RAGAS evaluation:

```python
from src.evaluation import RAGEvaluator

evaluator = RAGEvaluator(
    nq_file_path="./data/nq-train-sample.jsonl",
    num_questions=50,
    num_docs=100
)

evaluator.run_full_evaluation()
```

### Metrics

| Metric | Target | Description |
|--------|--------|-------------|
| Faithfulness | >0.9 | No hallucinations |
| Answer Relevancy | >0.8 | Addresses the question |
| Context Precision | >0.8 | Well-ranked results |
| Context Recall | >0.7 | Retrieved enough info |
| Context Relevancy | >0.8 | Context matches query |

---

## 📁 Project Structure

```
intellirag/
├── src/
│   ├── embedding.py           # Vector store creation
│   ├── retriever.py           # Advanced retriever with reranking
│   ├── generator.py           # LLM response generation
│   ├── loader.py              # Document loader with OCR
│   ├── evaluation.py          # RAGAS evaluation pipeline
│   ├── explainability.py      # Visualization tools
│   └── test_data_loader.py    # NQ dataset loader
├── data/                      # Documents and datasets
├── venv/                      # Virtual environment
├── app.py                     # Main application
├── requirements.txt           # Dependencies
├── .env                       # Environment variables
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Configuration

### Chunking Strategy
```python
chunk_size = 800        # Tokens per chunk
chunk_overlap = 200     # Overlap between chunks
```

### Retrieval Settings
```python
k = 10                  # Initial retrieval
top_n = 5              # After reranking
```

### LLM Settings
```python
model = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo"
temperature = 0.1       # Low for factual answers
max_tokens = 512
```

---

## 🔧 Advanced Features

### Explainability

```python
from src.explainability import RAGExplainer

explainer = RAGExplainer()

# Visualize retrieval scores
plt = explainer.visualize_retrieval_scores(docs, scores, query, top_n=5)
plt.show()

# Get source attribution
attribution = explainer.create_source_attribution_map(answer, sources)
```

### OCR Support

Automatically handles scanned PDFs and images:

```python
from src.loader import load_files

# Supports: .pdf, .docx, .txt, .png, .jpg, .tiff
documents = load_files([
    "scanned_document.pdf",
    "image.png",
    "text.docx"
])
```

---

## 📦 Dependencies

Core libraries:
- `langchain` - RAG framework
- `langchain-openai` - LLM integration
- `langchain-community` - Document loaders
- `faiss-cpu` - Vector store
- `sentence-transformers` - Embeddings and reranking
- `ragas` - Evaluation metrics
- `pytesseract` - OCR engine
- `beautifulsoup4` - HTML parsing

See `requirements.txt` for complete list.

---

## 🎯 Use Cases

### Document Q&A
Build intelligent question-answering systems over your documents.

### Knowledge Base
Create searchable knowledge bases with semantic understanding.

### Research Assistant
Retrieve and synthesize information from multiple sources.

### Customer Support
Answer customer queries using your documentation.

---

## 🛠️ Development

### Setup Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Tesseract (for OCR)
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract

# Windows
# Download from: https://github.com/UB-Mannheim/tesseract/wiki
```

### Running Tests

```bash
# Test document loader
python -m src.test_data_loader

# Test complete pipeline
python -m src.evaluation
```

---

## 📝 Environment Variables

Create a `.env` file:

```bash
# Required
TOEGETHERAI_API_KEY=your_together_ai_api_key

# Optional
OPENAI_API_KEY=your_openai_api_key  # If using OpenAI instead
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Natural Questions dataset by Google AI
- RAGAS evaluation framework
- LangChain for RAG infrastructure
- Sentence Transformers for embeddings
- Together AI for LLM hosting

---

## 📬 Contact

For questions or support, please open an issue on GitHub.

---

## 🚦 Status Indicators

- 🟢 Production Ready: All metrics above targets
- 🟡 Good: Minor improvements needed
- 🔴 Needs Work: Significant tuning required

---

