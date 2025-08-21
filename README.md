# Multi-Source Knowledge Retrieval and Answer Generation System

Multi-Source Knowledge Retrieval and Answer Generation System is a Streamlit-based application that enables users to interactively ask questions about the content of uploaded documents (PDF, DOCX, TXT) or web URLs. It leverages Retrieval-Augmented Generation (RAG) techniques, vector embeddings, and LLMs to provide accurate, context-aware answers.

---

## Features

- **Multi-source RAG**: Query across multiple uploaded files or URLs.
- **Supported formats**: PDF, DOCX, TXT, and web pages.
- **Modern UI**: Built with Streamlit for an interactive chat experience.
- **Vector Search**: Uses FAISS and sentence-transformers for efficient retrieval.
- **LLM Integration**: Uses Together AI's Llama-3.1-8B-Instruct-Turbo for answer generation.
- **Session Memory**: Maintains chat history for context.

---

## Demo

![screenshot or gif here if available]

---

## Installation

1. **Clone the repository:**
   ```powershell
   git clone https://github.com/SachinMosambe/Web-RAG-Base-Chatbot.git
   cd Web-RAG-Base-Chatbot
   ```

2. **Install dependencies:**
   ```powershell
   pip install -r requirements.txt
   ```

3. **Set up environment variables:**
   - Create a `.env` file in the root directory with the following keys:
     ```env
     TOEGETHERAI_API_KEY=your_togetherai_api_key
     HUGGINGFACEHUB_API_TOKEN=your_huggingface_token
     ```

4. **(Optional) Install browser drivers for Selenium:**
   - For URL loading, [download the appropriate driver](https://selenium-python.readthedocs.io/installation.html#drivers) and ensure it is in your PATH.

---

## Usage

1. **Start the Streamlit app:**
   ```powershell
   streamlit run app.py
   ```

2. **Interact with the chatbot:**
   - Upload files or enter URLs in the sidebar.
   - Ask questions in the main chat window.
   - View answers and chat history.

---

## Project Structure

```
├── app.py                  # Streamlit app entry point
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (not committed)
├── data/                   # Uploaded files and vector store
│   └── vector_store/       # FAISS vector store (auto-generated)
├── src/                    # Core modules
│   ├── loader.py           # File/URL loading logic
│   ├── embedding.py        # Embedding and vector store creation
│   ├── retriever.py        # Vector search and retrieval
│   └── generator.py        # LLM-based answer generation
└── test.py                 # Test script for environment and embeddings
```

---

## How It Works

1. **Load Documents**: Upload files or enter URLs. Documents are loaded and parsed using LangChain loaders.
2. **Create Vector Store**: Documents are split into chunks and embedded using HuggingFace models. FAISS is used for fast similarity search.
3. **Ask Questions**: User queries are matched to relevant document chunks via vector search.
4. **Generate Answers**: The LLM (Llama-3.1-8B-Instruct-Turbo via Together AI) generates answers using the retrieved context.

---

## Requirements

- Python 3.8+
- [Together AI API Key](https://www.together.ai/)
- [HuggingFace API Token](https://huggingface.co/settings/tokens)
- Chrome/Firefox driver for Selenium (for URL loading)

---

## Troubleshooting

- **Selenium errors**: Ensure the correct browser driver is installed and in your PATH.
- **API errors**: Check your API keys in the `.env` file.
- **Vector store issues**: Delete the `data/vector_store/` folder to reset embeddings.

---


