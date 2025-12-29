"""
Clean Natural Questions (NQ) Dataset Loader
Simplified version with better error handling and structure
"""

import json
from bs4 import BeautifulSoup
from langchain.schema import Document
from typing import List, Dict


class NQDataLoader:
    """Clean loader for Natural Questions dataset"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.stats = {"loaded": 0, "skipped": 0, "errors": 0}
    
    def _parse_html_to_text(self, html_content: str) -> str:
        """Convert NQ HTML to plain text"""
        soup = BeautifulSoup(html_content, "html.parser")
        return soup.get_text(separator=" ", strip=True)
    
    def _extract_ground_truth(self, record: dict) -> str:
        """Extract ground truth answer from NQ record"""
        annotations = record.get("annotations", [])
        if not annotations:
            return None
        
        annotation = annotations[0]
        
        # Method 1: Try short answers with tokens
        short_answers = annotation.get("short_answers", [])
        if short_answers:
            tokens = record.get("document_tokens", [])
            if tokens:
                sa = short_answers[0]
                start = sa.get("start_token", -1)
                end = sa.get("end_token", -1)
                
                if start >= 0 and end > start and end <= len(tokens):
                    answer = " ".join([
                        tokens[i].get("token", "") 
                        for i in range(start, end)
                    ])
                    if answer.strip():
                        return answer.strip()
        
        # Method 2: Fallback to yes/no answer
        yes_no = annotation.get("yes_no_answer", "NONE")
        if yes_no in ["YES", "NO"]:
            return yes_no
        
        return None
    
    def load_for_ragas(self, max_samples: int = 50) -> Dict[str, List]:
        """
        Load NQ dataset in RAGAS format
        Returns: {"question": [], "ground_truth": [], "answer": [], "contexts": []}
        """
        print(f"\n📖 Loading NQ dataset for RAGAS evaluation...")
        print(f"   Source: {self.file_path}")
        print(f"   Max samples: {max_samples}")
        
        data = {
            "question": [],
            "ground_truth": [],
            "answer": [],
            "contexts": []
        }
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if len(data["question"]) >= max_samples:
                    break
                
                try:
                    record = json.loads(line)
                    
                    # Get question
                    question = record.get("question_text", "").strip()
                    if not question:
                        self.stats["skipped"] += 1
                        continue
                    
                    # Get ground truth
                    ground_truth = self._extract_ground_truth(record)
                    if not ground_truth:
                        self.stats["skipped"] += 1
                        continue
                    
                    # Add to dataset
                    data["question"].append(question)
                    data["ground_truth"].append(ground_truth)
                    data["answer"].append("")  # Filled by RAG
                    data["contexts"].append([])  # Filled by RAG
                    self.stats["loaded"] += 1
                    
                except json.JSONDecodeError:
                    self.stats["errors"] += 1
                except Exception as e:
                    self.stats["errors"] += 1
        
        self._print_stats("RAGAS evaluation")
        return data
    
    def load_for_vectorstore(self, max_docs: int = 100) -> List[Document]:
        """
        Load NQ documents for vector store
        Returns: List of LangChain Document objects
        """
        print(f"\n📚 Loading NQ documents for vector store...")
        print(f"   Source: {self.file_path}")
        print(f"   Max documents: {max_docs}")
        
        documents = []
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if len(documents) >= max_docs:
                    break
                
                try:
                    record = json.loads(line)
                    
                    # Parse HTML content
                    html_content = record.get("document_text", "")
                    if not html_content:
                        self.stats["skipped"] += 1
                        continue
                    
                    plain_text = self._parse_html_to_text(html_content)
                    if not plain_text or len(plain_text) < 50:
                        self.stats["skipped"] += 1
                        continue
                    
                    # Create document with metadata
                    doc = Document(
                        page_content=plain_text,
                        metadata={
                            "source": f"nq_doc_{line_num}",
                            "example_id": record.get("example_id", ""),
                            "document_url": record.get("document_url", ""),
                            "question": record.get("question_text", "")
                        }
                    )
                    documents.append(doc)
                    self.stats["loaded"] += 1
                    
                except json.JSONDecodeError:
                    self.stats["errors"] += 1
                except Exception as e:
                    self.stats["errors"] += 1
        
        self._print_stats("vector store")
        return documents
    
    def _print_stats(self, purpose: str):
        """Print loading statistics"""
        print(f"\n   ✅ Loaded: {self.stats['loaded']} records")
        print(f"   ⚠️  Skipped: {self.stats['skipped']} records (no valid data)")
        if self.stats['errors'] > 0:
            print(f"   ❌ Errors: {self.stats['errors']} records")
        print(f"   📊 Success rate: {self.stats['loaded']/(self.stats['loaded']+self.stats['skipped']+self.stats['errors'])*100:.1f}%")
        
        # Reset stats for next load
        self.stats = {"loaded": 0, "skipped": 0, "errors": 0}


# Simple usage functions for backward compatibility
def load_nq_for_ragas(file_path: str, max_samples: int = 50) -> Dict[str, List]:
    """Simple function to load NQ for RAGAS"""
    loader = NQDataLoader(file_path)
    return loader.load_for_ragas(max_samples)


def load_nq_documents_for_vectorstore(file_path: str, max_docs: int = 100) -> List[Document]:
    """Simple function to load NQ documents for vector store"""
    loader = NQDataLoader(file_path)
    return loader.load_for_vectorstore(max_docs)


# Example usage
if __name__ == "__main__":
    NQ_FILE = "./data/simplified-nq-train.jsonl"
    
    print("="*80)
    print(" Natural Questions Dataset Loader - Test")
    print("="*80)
    
    # Test 1: Load for RAGAS
    loader = NQDataLoader(NQ_FILE)
    ragas_data = loader.load_for_ragas(max_samples=10)
    
    if ragas_data["question"]:
        print(f"\n📝 Sample Question:")
        print(f"   Q: {ragas_data['question'][0]}")
        print(f"   A: {ragas_data['ground_truth'][0]}")
    
    # Test 2: Load for vector store
    loader = NQDataLoader(NQ_FILE)
    documents = loader.load_for_vectorstore(max_docs=50)
    
    if documents:
        print(f"\n📄 Sample Document:")
        print(f"   Content: {documents[0].page_content[:150]}...")
        print(f"   Metadata: {documents[0].metadata}")
    
    print("\n" + "="*80)