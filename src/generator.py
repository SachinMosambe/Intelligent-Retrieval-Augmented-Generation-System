from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

# Enhanced prompt template
prompt = PromptTemplate.from_template("""
You are a precise question-answering assistant. Answer based ONLY on the provided context.

Guidelines:
- Provide clear, concise answers
- If context lacks information, state: "I cannot find sufficient information in the given context"
- Be specific and factual
- Don't add external knowledge

Context:
{context}

Question: {question}

Answer:
""")

llm = ChatOpenAI(
    openai_api_key=os.getenv("TOEGETHERAI_API_KEY"),
    base_url="https://api.together.xyz/v1",
    model_name="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    temperature=0.1,
    max_tokens=512,
    top_p=0.9,
)

def generate_response_enhanced(retriever, query):
    """
    Generate response with source attribution
    """
    # Get reranked documents
    docs = retriever.retrieve_and_rerank(query, k=10, top_n=5)
    
    if not docs:
        return {
            "answer": "I couldn't find relevant information to answer this question.",
            "sources": []
        }
    
    # Combine context
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Generate response
    formatted_prompt = prompt.format(context=context, question=query)
    response = llm.invoke(formatted_prompt)
    
    # Extract unique sources
    sources = list(set([doc.metadata.get("source", "unknown") for doc in docs]))
    
    return {
        "answer": response.content,
        "sources": sources,
        "num_sources": len(sources)
    }
