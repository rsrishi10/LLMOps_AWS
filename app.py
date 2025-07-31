import json
import sys
import os
import boto3
import streamlit as st

from langchain_community.vectorstores import FAISS
from QASystem.ingestion import get_vector_store, data_ingestion, bedrock_embeddings
from QASystem.retrievalandgeneration import get_response_llm, get_llm

def main():
    st.set_page_config(page_title="QA with Document")
    st.header("QA with Document using Langchain")
    
    user_question = st.text_input("Enter your question from pdf file", key="user_input")
    
    with st.sidebar:
        st.title("update or create the vector store")
        if st.button("vector update"):
            with st.spinner("processing..."):
                docs = data_ingestion()
                get_vector_store(docs)
                st.success("vector store updated successfully")
        
        if st.button("llama model"):
            if not user_question:
                st.error("Please enter a question first!")
                return
                
            with st.spinner("processing..."):
                faiss_index = FAISS.load_local(
                    "faiss_index", 
                    bedrock_embeddings,
                    allow_dangerous_deserialization=True  
                )
                llm = get_llm()
                
                response = get_response_llm(llm, faiss_index, user_question)
                st.write("Answer:", response["result"])
                st.success("Done")

if __name__ == "__main__":
    main()

