"""
Clean RAG Evaluation using RAGAS
Simplified structure with better error handling
"""

import os
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    context_relevancy
)

# Import your RAG components
from embedding import create_vector_store_enhanced
from generator import generate_response_enhanced

# Import data loader
from test_data_loader import NQDataLoader

load_dotenv()


class RAGEvaluator:
    """Clean evaluator for RAG systems using RAGAS"""
    
    def __init__(self, nq_file_path: str, num_questions: int = 50, num_docs: int = 100):
        self.nq_file_path = nq_file_path
        self.num_questions = num_questions
        self.num_docs = num_docs
        self.retriever = None
        
        # RAGAS metrics
        self.metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
            context_relevancy
        ]
    
    def setup_rag_system(self):
        """Initialize RAG system with vector store and retriever"""
        print("\n🔧 Setting up RAG system...")
        
        # Load documents for knowledge base
        print(f"\n[1/2] Loading {self.num_docs} documents for knowledge base...")
        loader = NQDataLoader(self.nq_file_path)
        documents = loader.load_for_vectorstore(max_docs=self.num_docs)
        
        if not documents:
            raise ValueError("❌ No documents loaded for knowledge base")
        
        # Create vector store
        print(f"\n[2/2] Creating vector store with enhanced settings...")
        print("   • Optimized chunking (800 tokens, 200 overlap)")
        print("   • Normalized embeddings")
        print("   • Cross-encoder reranking enabled")
        
        vector_store = create_vector_store_enhanced(documents)
        from src.retriever import search_vector_store_enhanced
        self.retriever = search_vector_store_enhanced(vector_store)
        
        print("   ✅ RAG system ready!")
    
    def run_rag_on_questions(self, data: dict) -> dict:
        """Run RAG system on test questions and collect results"""
        print(f"\n🤖 Running RAG on {len(data['question'])} questions...")
        
        processed = 0
        failed = 0
        
        for i, question in enumerate(data["question"]):
            try:
                # Retrieve and rerank
                retrieved_docs = self.retriever.retrieve_and_rerank(
                    question, 
                    k=10, 
                    top_n=5
                )
                
                # Extract contexts
                contexts = [doc.page_content for doc in retrieved_docs]
                data["contexts"][i] = contexts if contexts else [""]
                
                # Generate answer
                result = generate_response_enhanced(self.retriever, question)
                
                # Extract answer
                if isinstance(result, dict):
                    data["answer"][i] = result.get("answer", "")
                else:
                    data["answer"][i] = str(result)
                
                processed += 1
                
                if (i + 1) % 10 == 0:
                    print(f"   ✓ Processed {i + 1}/{len(data['question'])}")
                
            except Exception as e:
                print(f"   ⚠️  Error on Q{i+1}: {str(e)[:50]}")
                data["answer"][i] = "ERROR"
                data["contexts"][i] = [""]
                failed += 1
        
        print(f"\n   ✅ Completed: {processed} successful, {failed} failed")
        return data
    
    def filter_valid_samples(self, data: dict) -> dict:
        """Remove failed samples before evaluation"""
        filtered = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": []
        }
        
        for i in range(len(data["question"])):
            if data["answer"][i] != "ERROR" and data["contexts"][i] != [""]:
                filtered["question"].append(data["question"][i])
                filtered["answer"].append(data["answer"][i])
                filtered["contexts"].append(data["contexts"][i])
                filtered["ground_truth"].append(data["ground_truth"][i])
        
        print(f"\n📊 Valid samples: {len(filtered['question'])}/{len(data['question'])}")
        return filtered
    
    def evaluate_with_ragas(self, data: dict):
        """Run RAGAS evaluation"""
        print("\n" + "="*80)
        print(" Running RAGAS Evaluation")
        print("="*80)
        
        # Filter valid samples
        data = self.filter_valid_samples(data)
        
        if len(data["question"]) == 0:
            print("❌ No valid samples to evaluate!")
            return None
        
        # Convert to dataset
        dataset = Dataset.from_dict(data)
        
        # Show metrics
        print("\n📋 Metrics:")
        print("   1. Faithfulness - No hallucinations")
        print("   2. Answer Relevancy - Answers the question")
        print("   3. Context Precision - Good ranking")
        print("   4. Context Recall - Retrieved enough info")
        print("   5. Context Relevancy - Context matches query")
        
        print("\n⏳ Evaluating (this may take a few minutes)...")
        
        # Run evaluation
        results = evaluate(dataset=dataset, metrics=self.metrics)
        
        return results
    
    def display_results(self, results):
        """Display evaluation results clearly"""
        print("\n" + "="*80)
        print(" EVALUATION RESULTS")
        print("="*80)
        
        df = results.to_pandas()
        
        print("\n📊 Overall Scores:")
        print("-" * 80)
        
        # Metric definitions
        metrics_info = {
            "faithfulness": ("Faithfulness", 0.9, "No hallucinations"),
            "answer_relevancy": ("Answer Relevancy", 0.8, "Relevant to question"),
            "context_precision": ("Context Precision", 0.8, "Good ranking"),
            "context_recall": ("Context Recall", 0.7, "Retrieved enough"),
            "context_relevancy": ("Context Relevancy", 0.8, "Context matches query")
        }
        
        scores = {}
        for metric_key, (name, target, desc) in metrics_info.items():
            if metric_key in df.columns:
                score = df[metric_key].mean()
                scores[metric_key] = score
                
                # Status indicator
                if score >= target:
                    status = "🟢"
                elif score >= target - 0.1:
                    status = "🟡"
                else:
                    status = "🔴"
                
                print(f"{status} {name:<22} {score:.3f}  (Target: >{target:.1f})")
                print(f"   └─ {desc}")
        
        # Overall assessment
        print("\n🎯 System Assessment:")
        print("-" * 80)
        
        avg_score = sum(scores.values()) / len(scores) if scores else 0
        faithfulness_score = scores.get("faithfulness", 0)
        
        if faithfulness_score >= 0.9 and avg_score >= 0.8:
            print("  Status: 🟢 PRODUCTION READY")
            print("  Your RAG system meets industry standards!")
        elif faithfulness_score >= 0.85 and avg_score >= 0.7:
            print("  Status: 🟡 GOOD (Minor improvements needed)")
        else:
            print("  Status: 🔴 NEEDS IMPROVEMENT")
            print("  Tip: Tune chunk size, improve retrieval, or refine prompts")
        
        # Sample results
        print("\n📋 Sample Results:")
        print("-" * 80)
        
        for i in range(min(3, len(df))):
            row = df.iloc[i]
            print(f"\n{i+1}. Q: {row.get('question', '')[:60]}...")
            print(f"   A: {row.get('answer', '')[:60]}...")
            if 'faithfulness' in row:
                print(f"   Faithfulness: {row['faithfulness']:.2f} | " + 
                      f"Relevancy: {row.get('answer_relevancy', 0):.2f}")
        
        # Save results
        output_file = "ragas_evaluation_results.csv"
        df.to_csv(output_file, index=False)
        print(f"\n💾 Results saved to: {output_file}")
        print("="*80)
    
    def run_full_evaluation(self):
        """Run complete evaluation pipeline"""
        print("\n" + "="*80)
        print(" 🚀 RAG Evaluation with RAGAS")
        print("="*80)
        
        # Check file exists
        if not os.path.exists(self.nq_file_path):
            print(f"\n❌ Dataset not found: {self.nq_file_path}")
            print("Download from: https://ai.google.com/research/NaturalQuestions")
            return
        
        # Step 1: Load test questions
        print(f"\n[Step 1/4] Loading {self.num_questions} test questions...")
        loader = NQDataLoader(self.nq_file_path)
        data = loader.load_for_ragas(max_samples=self.num_questions)
        
        if len(data["question"]) == 0:
            print("❌ No questions loaded!")
            return
        
        # Step 2: Setup RAG system
        print(f"\n[Step 2/4] Setting up RAG system...")
        self.setup_rag_system()
        
        # Step 3: Run RAG
        print(f"\n[Step 3/4] Running RAG on questions...")
        data = self.run_rag_on_questions(data)
        
        # Step 4: Evaluate
        print(f"\n[Step 4/4] Evaluating with RAGAS...")
        results = self.evaluate_with_ragas(data)
        
        if results:
            self.display_results(results)
            print("\n✅ Evaluation complete!")
        else:
            print("\n❌ Evaluation failed!")


def main():
    """Main entry point"""
    # Configuration
    NQ_FILE = "./data/nq-train-sample.jsonl"
    NUM_QUESTIONS = 50
    NUM_DOCS = 100
    
    # Run evaluation
    evaluator = RAGEvaluator(
        nq_file_path=NQ_FILE,
        num_questions=NUM_QUESTIONS,
        num_docs=NUM_DOCS
    )
    
    evaluator.run_full_evaluation()


if __name__ == "__main__":
    main()