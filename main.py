import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
# Document Loaders
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader
)

loader = PyPDFLoader("Muhammad_Hamza_Resume.pdf")
document = loader.load()

# Text Splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter = RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=20)
chunks = splitter.split_documents(document)

# Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI
model = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

# Vector Database
from langchain_chroma import Chroma
vector_store = Chroma(
    collection_name="pdf",
    embedding_function=embeddings,
    persist_directory="chroma_db"
)

vector_store.add_documents(chunks)

# Retriever
retriever = vector_store.as_retriever(
    search_type="mmr",                   # <-- This enables MMR
    search_kwargs={"k": 3, "lambda_mult": 0.5}  # k = top results, lambda_mult = relevance-diversity balance
)

query = input("Ask Question: ")
results = retriever.invoke(query)
# for i, doc in enumerate(results):
#     print(f"\n--- Result {i+1} ---")
#     print(doc.page_content)

# Prompt
from langchain_core.prompts import PromptTemplate
template = PromptTemplate(
    template="""
You are a helpful assistant.

Answer the user's question using only the provided context.

Context:
{context}

Question:
{question}

If the answer is not available in the context, say:
"I don't know based on the provided document."

Answer:
""",
input_variables=["context","question"]                         
)

prompt = template.invoke({"context": results, "question": query})

# LLM
llm_response = model.invoke(prompt)

# Result
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
final_result = parser.invoke(llm_response)
print(final_result)