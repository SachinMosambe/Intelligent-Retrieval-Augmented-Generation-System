from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

prompt = PromptTemplate.from_template(""" 

    You are an intelligent assistant. Use the provided context to answer the user's question. 
    If the answer is not in the context, reply: "Sorry, I couldn't find the answer in the given information."

    Context: {context}
    Question: {question}
    
    answer:

"""
)

llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.1",
        huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
        temperature =0.3, 
        max_new_tokens = 512
    )

def generate_response(retriever, query):
    
    qa_chain = RetrievalQA.from_chain_type(
        llm = llm,
        retriever = retriever,
        chain_type = "stuff",
        chain_type_kwargs = {"prompt": prompt},
        return_source_documents =True
    )

    result = qa_chain.invoke({"query":query})
    return result["result"]