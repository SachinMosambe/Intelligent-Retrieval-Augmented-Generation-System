from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
import numpy as np

class EnhancedRetriever:
    """
    Advanced retriever with reranking and query expansion
    """
    def __init__(self, vector_store):
        self.vector_store = vector_store
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def expand_query(self, query):
        """
        Query expansion to handle ambiguity
        """
        # Simple expansion with synonyms/variations
        expanded_queries = [query]
        
        # Add variations for common patterns
        if "?" in query:
            expanded_queries.append(query.replace("?", ""))
        
        # Add lowercase version
        if query != query.lower():
            expanded_queries.append(query.lower())
        
        return expanded_queries
    
    def retrieve_and_rerank(self, query, k=10, top_n=5):
        """
        Retrieve with query expansion and rerank
        """
        # Expand query to handle ambiguity
        queries = self.expand_query(query)
        
        all_docs = []
        seen = set()
        
        # Retrieve from all query variations
        for q in queries:
            docs = self.vector_store.similarity_search(q, k=k)
            for doc in docs:
                doc_id = doc.page_content[:100]  # Use first 100 chars as ID
                if doc_id not in seen:
                    all_docs.append(doc)
                    seen.add(doc_id)
        
        if not all_docs:
            return []
        
        # Rerank with cross-encoder
        pairs = [[query, doc.page_content] for doc in all_docs]
        scores = self.reranker.predict(pairs)
        
        # Sort and return top_n
        scored_docs = list(zip(all_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs[:top_n]]
    
    def get_relevant_documents(self, query):
        """
        Compatibility method for LangChain
        """
        return self.retrieve_and_rerank(query)

def search_vector_store_enhanced(vector_store):
    """
    Create enhanced retriever
    """
    return EnhancedRetriever(vector_store)
