from langchain_openai import ChatOpenAI
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

llm = ChatOpenAI(
    openai_api_key = os.getenv("TOEGETHERAI_API_KEY"),
    base_url= "https://api.together.xyz/v1",
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    temperature = 0.1,
    max_tokens = 512,
    top_p = 0.9,
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