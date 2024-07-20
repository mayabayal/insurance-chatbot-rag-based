import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

# Load the GROQ and OpenAI API keys
groq_api_key = os.getenv('GROQ_API_KEY')
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.title("Gemma Model Document Q&A")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

def initialize_session_state():
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = None
    if "loader" not in st.session_state:
        st.session_state.loader = None
    if "docs" not in st.session_state:
        st.session_state.docs = None
    if "text_splitter" not in st.session_state:
        st.session_state.text_splitter = None
    if "final_documents" not in st.session_state:
        st.session_state.final_documents = None
    if "vectors" not in st.session_state:
        st.session_state.vectors = None

initialize_session_state()

def vector_embedding():
    if not st.session_state.vectors:
        st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        # Corrected Path to PDF Directory
        pdf_directory = "C:\\Users\\Subhash\\OneDrive\\Desktop\\GENAI\\GEMMA\\pdf_downloads"
        
        # Check if directory exists
        if not os.path.exists(pdf_directory):
            st.error(f"Directory does not exist: {pdf_directory}")
            return
        
        # Check if directory contains PDF files
        pdf_files = [file for file in os.listdir(pdf_directory) if file.endswith(".pdf")]
        
        if not pdf_files:
            st.error(f"No PDF files found in directory: {pdf_directory}")
            return
        
        # Debugging: Print the list of PDF files found
        st.write(f"PDF files found: {pdf_files}")
        
        # Data Ingestion
        st.session_state.loader = PyPDFDirectoryLoader(pdf_directory)
        st.session_state.docs = st.session_state.loader.load()
        
        # Debugging: Check the number of documents loaded
        st.write(f"Number of documents loaded: {len(st.session_state.docs)}")
        
        if len(st.session_state.docs) == 0:
            st.error("No documents found. Please check the directory and try again.")
            return

        # Chunk Creation
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        
        # Debugging: Check the number of chunks created
        st.write(f"Number of chunks created: {len(st.session_state.final_documents)}")
        
        if len(st.session_state.final_documents) == 0:
            st.error("No document chunks created. Please check the splitting logic.")
            return
        
        # Vector Embedding
        try:
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
            st.write("Vector Store DB is ready")
        except IndexError as e:
            st.error(f"Error creating vector store: {e}")
            st.write("Check if the embeddings are being generated correctly.")

prompt1 = st.text_input("Enter Your Question From Documents")

if st.button("Documents Embedding"):
    vector_embedding()

import time

if prompt1:
    if not st.session_state.vectors:
        st.error("Vector Store is not ready. Please embed the documents first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.write("Response time: ", time.process_time() - start)
        st.write(response['answer'])

        # With a streamlit expander
        with st.expander("Document Similarity Search"):
            # Find the relevant chunks
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("--------------------------------")
