import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class RAGExplainer:
    """
    Explainability tools for RAG system
    """
    
    @staticmethod
    def visualize_retrieval_scores(docs, scores, query, top_n=5):
        """
        Visualize retrieval scores for transparency
        """
        plt.figure(figsize=(10, 6))
        
        doc_labels = [f"Doc {i+1}" for i in range(min(top_n, len(docs)))]
        top_scores = scores[:top_n]
        
        sns.barplot(x=top_scores, y=doc_labels, palette="viridis")
        plt.xlabel("Relevance Score")
        plt.ylabel("Document")
        plt.title(f"Top {top_n} Retrieved Documents for: '{query}'")
        plt.tight_layout()
        
        return plt
    
    @staticmethod
    def create_source_attribution_map(answer, sources):
        """
        Map answer sentences to sources
        """
        attribution = {
            "answer": answer,
            "sources": sources,
            "attribution_map": {
                "sentence_1": sources[0] if sources else "unknown",
                "sentence_2": sources[1] if len(sources) > 1 else sources[0] if sources else "unknown"
            }
        }
        return attribution
    
    @staticmethod
    def explain_retrieval(query, docs, scores):
        """
        Provide human-readable explanation of retrieval
        """
        explanation = f"Query: '{query}'\n\n"
        explanation += f"Retrieved {len(docs)} documents:\n"
        
        for i, (doc, score) in enumerate(zip(docs[:3], scores[:3])):
            explanation += f"\n{i+1}. Score: {score:.3f}\n"
            explanation += f"   Source: {doc.metadata.get('source', 'unknown')}\n"
            explanation += f"   Preview: {doc.page_content[:100]}...\n"
        
        return explanation
