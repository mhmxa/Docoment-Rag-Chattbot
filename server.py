"""
FastAPI wrapper around pdf_rag_chatbot.py

Exposes two HTTP endpoints for the Folio frontend (index.html):
  POST /upload  -> multipart 'file' (a PDF)         -> { session_id, pages, chunks }
  POST /chat    -> JSON { question, session_id }     -> { answer, sources: [{source, page}] }

Run it with:
    pip install fastapi uvicorn python-multipart
    uvicorn server:app --reload --port 8000

Then serve index.html separately, e.g.:
    python -m http.server 5500
and open http://localhost:5500 in your browser.
"""

import os
import shutil
import tempfile
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.output_parsers import StrOutputParser

# Reuse everything already built and tested in pdf_rag_chatbot.py
from main import (
    validate_pdf,
    load_pdf,
    split_documents,
    get_embeddings,
    get_vector_store,
    get_model,
    create_auto_retriever,
    get_universal_prompt,
    sanitize_input,
    QueryCache,
)

app = FastAPI(title="Folio RAG API")

# Allow the browser-based frontend (served from any local port, or opened as a
# file) to call this API. Tighten allow_origins to your real frontend URL
# before deploying anywhere public.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store: session_id -> {retriever, model, prompt, chain, cache, history}
SESSIONS: dict = {}

# Shared, lazily-created resources that don't need to be rebuilt per session
_embeddings = None
_model = None


def _get_shared_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = get_embeddings()
        if _embeddings is None:
            raise HTTPException(status_code=500, detail="No embeddings backend available on the server.")
    return _embeddings


def _get_shared_model():
    global _model
    if _model is None:
        _model = get_model()
    return _model


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None


@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(SESSIONS)}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Persist the upload to a temp file so the existing loader (which expects
    # a file path) can read it.
    tmp_dir = tempfile.mkdtemp(prefix="folio_upload_")
    tmp_path = os.path.join(tmp_dir, file.filename)
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    if not validate_pdf(tmp_path):
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Uploaded file failed PDF validation.")

    documents = load_pdf(tmp_path)
    if not documents:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail="Could not extract any content from the PDF.")

    chunks = split_documents(documents)

    embeddings = _get_shared_embeddings()
    model = _get_shared_model()

    session_id = str(uuid.uuid4())
    persist_dir = os.path.join(tempfile.gettempdir(), "folio_chroma", session_id)

    try:
        vector_store = get_vector_store(embeddings, chunks, persist_dir=persist_dir)
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Error building vector store: {e}")

    retriever = create_auto_retriever(vector_store, model, chunks)
    prompt = get_universal_prompt()

    SESSIONS[session_id] = {
        "retriever": retriever,
        "model": model,
        "prompt": prompt,
        "cache": QueryCache(),
        "history": [],
        "filename": file.filename,
        "pages": len(documents),
        "chunks": len(chunks),
        "tmp_dir": tmp_dir,
    }

    return {
        "session_id": session_id,
        "pages": len(documents),
        "chunks": len(chunks),
        "filename": file.filename,
    }


@app.post("/chat")
async def chat(req: ChatRequest):
    if not req.session_id or req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Unknown or missing session_id. Upload a document first.")

    session = SESSIONS[req.session_id]
    question = sanitize_input(req.question.strip())
    if not question:
        raise HTTPException(status_code=400, detail="Empty question.")

    cached = session["cache"].get(question)
    if cached:
        return cached

    retriever = session["retriever"]
    model = session["model"]
    prompt = session["prompt"]

    try:
        docs = retriever.invoke(question)
    except AttributeError:
        # Older retriever interfaces
        docs = retriever.get_relevant_documents(question)

    context = "\n\n".join(d.page_content for d in docs)
    chain = prompt | model | StrOutputParser()

    try:
        answer = chain.invoke({"context": context, "question": question})
    except Exception as e:
        msg = str(e)
        if "429" in msg or "quota" in msg.lower():
            raise HTTPException(status_code=429, detail="Model API quota exceeded. Please try again shortly.")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {e}")

    sources = []
    seen = set()
    for d in docs:
        key = (d.metadata.get("source"), d.metadata.get("page"))
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source": d.metadata.get("source"), "page": d.metadata.get("page")})

    result = {"answer": answer, "sources": sources}
    session["cache"].set(question, result)
    session["history"].append((question, answer))
    return result


@app.delete("/session/{session_id}")
def end_session(session_id: str):
    session = SESSIONS.pop(session_id, None)
    if session and session.get("tmp_dir"):
        shutil.rmtree(session["tmp_dir"], ignore_errors=True)
    return {"ended": session_id}
