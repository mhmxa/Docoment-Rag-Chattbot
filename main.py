import os
import time
import hashlib
import random
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

# -------------------------
# 1. ERROR HANDLING & VALIDATION
# -------------------------
def validate_pdf(file_path):
    """Check if PDF file exists and is valid"""
    if not file_path:
        print("❌ No file path provided")
        return False
    
    if not os.path.exists(file_path):
        print(f"❌ File not found: {file_path}")
        return False
    
    if not file_path.lower().endswith('.pdf'):
        print(f"❌ Not a PDF file: {file_path}")
        return False
    
    if os.path.getsize(file_path) == 0:
        print(f"❌ File is empty: {file_path}")
        return False
    
    return True

# -------------------------
# 2. FILE MANAGEMENT
# -------------------------
from langchain_community.document_loaders import PyPDFLoader

def load_pdf(file_path):
    """Load PDF with error handling and metadata"""
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        
        # Add source info to each page
        for doc in documents:
            doc.metadata['source'] = os.path.basename(file_path)
            doc.metadata['page'] = doc.metadata.get('page', 0)
        
        print(f"✅ Loaded {len(documents)} pages from {os.path.basename(file_path)}")
        return documents
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        return []

# -------------------------
# 3. CHUNKING OPTIMIZATION
# -------------------------
from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(documents):
    """Split documents with appropriate chunk size based on document length"""
    total_pages = len(documents)
    total_text = sum(len(doc.page_content) for doc in documents)
    
    print(f"📊 Document stats: {total_pages} pages, {total_text} characters")
    
    # Adjust chunk size based on document size
    if total_pages < 5:
        chunk_size = 300
        overlap = 30
    elif total_pages < 20:
        chunk_size = 500
        overlap = 50
    elif total_pages < 50:
        chunk_size = 700
        overlap = 70
    else:
        chunk_size = 1000
        overlap = 100
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks (size: {chunk_size}, overlap: {overlap})")
    return chunks

# -------------------------
# 4. VECTOR STORE
# -------------------------
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

def add_documents_with_retry(vector_store, chunks, max_retries=3):
    """Add documents with automatic retry on quota errors"""
    for attempt in range(max_retries):
        try:
            for chunk in chunks:
                content_hash = hashlib.md5(chunk.page_content.encode()).hexdigest()
                chunk.metadata['content_hash'] = content_hash
            
            vector_store.add_documents(chunks)
            print(f"✅ Added {len(chunks)} chunks to vector store")
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "quota" in error_msg.lower():
                import re
                retry_time = 60
                retry_match = re.search(r'retry in (\d+\.?\d*)s', error_msg)
                if retry_match:
                    retry_time = float(retry_match.group(1)) + 5
                
                print(f"⚠️ API quota exceeded. Waiting {retry_time:.0f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(retry_time)
                time.sleep(random.uniform(1, 3))
            else:
                print(f"❌ Error adding documents: {e}")
                return False
    
    print("❌ Failed to add documents after multiple retries")
    return False

def get_vector_store(embeddings, chunks=None, persist_dir="chroma_db"):
    """Get vector store with deduplication"""
    vector_store = Chroma(
        collection_name="pdf_collection",
        embedding_function=embeddings,
        persist_directory=persist_dir
    )
    
    if chunks and vector_store._collection.count() == 0:
        success = add_documents_with_retry(vector_store, chunks)
        if not success:
            print("⚠️ Could not add all documents")
    
    return vector_store

# -------------------------
# 5. AUTO RETRIEVER SELECTION
# -------------------------
def create_auto_retriever(vector_store, model, chunks):
    """Automatically select the best retriever based on document size"""
    
    num_chunks = len(chunks)
    print(f"\n📊 Auto-selecting retriever for {num_chunks} chunks...")
    
    # For small documents (<= 20 chunks): Use simple similarity
    if num_chunks <= 20:
        print("   → Using: Basic similarity (small document)")
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )
    
    # For medium documents (21-100 chunks): Use similarity with threshold
    elif num_chunks <= 100:
        print("   → Using: Similarity with confidence threshold (medium document)")
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 4,
                "score_threshold": 0.3
            }
        )
    
    # For large documents (101+ chunks): Use multi-query for better coverage
    else:
        print("   → Using: Multi-query retriever (large document)")
        from langchain_community.retrievers import MultiQueryRetriever
        
        retriever = MultiQueryRetriever.from_llm(
            retriever=vector_store.as_retriever(
                search_kwargs={"k": 5}
            ),
            llm=model,
            include_original=True,
        )
    
    return retriever

# -------------------------
# 6. UNIVERSAL PROMPT
# -------------------------
from langchain_core.prompts import PromptTemplate

def get_universal_prompt():
    """Get a universal prompt that works for all questions"""
    template = """You are a helpful AI assistant. Answer the user's question using ONLY the information provided in the context below.

Context:
{context}

Question: {question}

Instructions:
1. Use ONLY the information from the context
2. If the answer is not in the context, say "I don't have enough information to answer that."
3. Be concise but thorough
4. If citing specific information, mention the source if available

Answer:"""
    
    return PromptTemplate(template=template, input_variables=["context", "question"])

# -------------------------
# 7. MODEL SETUP
# -------------------------
from langchain_google_genai import ChatGoogleGenerativeAI

def get_model(retry_count=0):
    """Get model with retry logic"""
    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        print("✅ Using model: gemini-3.6-flash")
        return model
    except Exception as e:
        if "429" in str(e) or "quota" in str(e).lower():
            if retry_count < 3:
                wait_time = 60 * (retry_count + 1)
                print(f"⚠️ API quota exceeded. Waiting {wait_time} seconds...")
                time.sleep(wait_time)
                return get_model(retry_count + 1)
            else:
                print("❌ API quota exceeded. Please try again later.")
                raise
        else:
            raise

# -------------------------
# 8. EMBEDDINGS
# -------------------------
def get_embeddings():
    """Get embeddings with fallback"""
    try:
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        if os.getenv("GOOGLE_API_KEY"):
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            print("✅ Using Google embeddings")
            return embeddings
    except:
        pass
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        print("✅ Using Hugging Face embeddings")
        return embeddings
    except:
        print("❌ No embeddings available. Please install sentence-transformers")
        return None

# -------------------------
# 9. QUERY CACHE
# -------------------------
class QueryCache:
    def __init__(self, max_size=100):
        self.cache = {}
        self.max_size = max_size
    
    def get(self, query):
        key = hashlib.md5(query.encode()).hexdigest()
        if key in self.cache:
            timestamp, response = self.cache[key]
            if time.time() - timestamp < 3600:
                return response
            else:
                del self.cache[key]
        return None
    
    def set(self, query, response):
        key = hashlib.md5(query.encode()).hexdigest()
        self.cache[key] = (time.time(), response)
        if len(self.cache) > self.max_size:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][0])
            del self.cache[oldest_key]

# -------------------------
# 10. SANITIZE INPUT
# -------------------------
def sanitize_input(text):
    text = text.replace('<', '').replace('>', '').replace('{', '').replace('}', '')
    if len(text) > 5000:
        text = text[:5000]
    return text

# -------------------------
# 11. MAIN APPLICATION
# -------------------------
def main():
    print("="*50)
    print("📚 PDF RAG Chatbot (Automatic Version)")
    print("="*50)
    
    # Setup
    cache = QueryCache()
    chat_history = []
    
    # 1. Get PDF file
    pdf_path = input("📄 Enter PDF path: ").strip()
    if not validate_pdf(pdf_path):
        return
    
    # 2. Load PDF
    documents = load_pdf(pdf_path)
    if not documents:
        return
    
    # 3. Split documents
    chunks = split_documents(documents)
    
    # 4. Setup embeddings
    embeddings = get_embeddings()
    if embeddings is None:
        print("❌ Could not load embeddings. Please install sentence-transformers")
        print("   Run: pip install sentence-transformers")
        return
    
    # 5. Setup vector store
    try:
        vector_store = get_vector_store(embeddings, chunks)
    except Exception as e:
        print(f"❌ Error setting up vector store: {e}")
        return
    
    # 6. Setup model
    try:
        model = get_model()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # 7. Auto-select retriever (no user input needed!)
    retriever = create_auto_retriever(vector_store, model, chunks)
    
    # 8. Use universal prompt (no user input needed!)
    prompt = get_universal_prompt()
    print("✅ Using universal prompt")
    
    # 9. Create RAG chain
    from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
    from langchain_core.output_parsers import StrOutputParser
    
    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)
    
    rag_chain = (
        RunnableParallel(
            context=retriever | RunnableLambda(format_docs),
            question=RunnablePassthrough()
        )
        | prompt
        | model
        | StrOutputParser()
    )
    
    print("\n" + "="*50)
    print("✅ Chatbot is ready!")
    print("Commands:")
    print("  - Type your question to ask")
    print("  - 'history' to see chat history")
    print("  - 'clear' to clear history")
    print("  - 'exit' to quit")
    print("="*50)
    
    # 10. Interactive chat loop
    while True:
        query = input("\n💬 You: ").strip()
        
        if query.lower() == 'exit':
            print("👋 Goodbye!")
            break
        
        if query.lower() == 'history':
            if not chat_history:
                print("No history yet")
            else:
                print("\n📜 Chat History:")
                for i, (q, a) in enumerate(chat_history[-5:], 1):
                    print(f"{i}. You: {q[:50]}...")
                    print(f"   Bot: {a[:50]}...")
                    print()
            continue
        
        if query.lower() == 'clear':
            chat_history = []
            print("✅ History cleared")
            continue
        
        # Sanitize input
        query = sanitize_input(query)
        
        # Check cache
        cached_response = cache.get(query)
        if cached_response:
            print(f"\n🤖 (from cache): {cached_response}")
            continue
        
        try:
            print("⏳ Thinking...")
            start_time = time.time()
            
            response = rag_chain.invoke(query)
            
            end_time = time.time()
            
            print(f"\n🤖 Answer: {response}")
            print(f"⏱️ Time: {end_time - start_time:.2f}s")
            
            cache.set(query, response)
            chat_history.append((query, response))
            
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                print(f"❌ API Quota exceeded. Please wait a few minutes.")
            else:
                print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()