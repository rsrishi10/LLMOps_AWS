from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.llms import Bedrock
import boto3
import json
import os

## bedrock client
bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1" 
)

bedrock_embeddings = BedrockEmbeddings(
    credentials_profile_name=None,  
    region_name="us-east-1", 
    model_id="amazon.titan-embed-text-v2:0"
)

def data_ingestion():
    loader = PyPDFDirectoryLoader("data")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=1000)
    docs = text_splitter.split_documents(documents)
    
    return docs

def get_vector_store(docs):
    vector_store_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vector_store_faiss.save_local("faiss_index")
    return vector_store_faiss

if __name__ == "__main__":
    docs = data_ingestion()
    get_vector_store(docs)



