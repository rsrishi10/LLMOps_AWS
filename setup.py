from setuptools import setup, find_packages

setup(
    name="QASystem", 
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "boto3>=1.34.131",
        "langchain",
        "langchain_aws>=0.2.7",
        "pypdf",
        "faiss-cpu",
        "streamlit"
    ]
)