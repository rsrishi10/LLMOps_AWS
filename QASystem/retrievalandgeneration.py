from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_aws import BedrockLLM
import boto3
from QASystem.ingestion import get_vector_store
from QASystem.ingestion import data_ingestion, bedrock_embeddings

bedrock_runtime = boto3.client(
    service_name="bedrock-runtime",
    region_name="us-east-1" 
)

def get_llm():
    llm = BedrockLLM(
        model_id="meta.llama3-8b-instruct-v1:0",
        model_kwargs={
            "max_tokens": 512,
            "temperature": 0.0,
            "top_p": 0.9
        },
        region_name="us-east-1"
    )
    return llm

prompt_template = """
    Human: Use the following pieces of a context to provide a 
    concise answer to the question. If you don't know the answer, just say that you don't know.
    <context>
    {context}
    </context>
    Question: {question}

    Assistant: Let me help you with that question based on the provided context.
    """ 

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context","question"]
)

def get_response_llm(llm, vectorstore_faiss, query):
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore_faiss.as_retriever(
            search_type="similarity",
            search_kwargs={"k":3}
            ),
        return_source_documents=True,
        chain_type_kwargs={"prompt":PROMPT}
    )
    
    answer = qa.invoke({"query": query})
    return answer

def load_vector_store():
    from langchain_community.vectorstores import FAISS
    return FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)

if __name__ == "__main__":
    docs = data_ingestion()
    vectorstore_faiss = get_vector_store(docs)
    query = "What is RAG token?"
    llm = get_llm()
    response = get_response_llm(llm, vectorstore_faiss, query)
    print("\nQuestion:", query)
    print("\nAnswer:", response["result"])
    print("\nSources:")
    for doc in response["source_documents"]:
        print("\n- Page Content:", doc.page_content[:200], "...")
    
  
    

    