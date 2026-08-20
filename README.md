# Folio — PDF RAG Chatbot

Ask questions about any PDF and get answers grounded in the document itself, with
page-level citations. Upload a PDF through the web interface, and the app chunks,
embeds, and indexes it, then answers your questions using only that document's
content.

## How it works

1. **`main.py`** — the core RAG pipeline: loads and chunks a PDF, embeds it into a
   Chroma vector store, auto-selects a retriever based on document size, and
   answers questions through a Gemini model via LangChain.
2. **`server.py`** — a FastAPI wrapper around `main.py` that exposes it over HTTP
   (`/upload`, `/chat`) so a browser can use it, with CORS enabled.
3. **`index.html`** — the frontend (Folio): drag-and-drop PDF upload, a processing
   animation, and a chat interface with citation chips. Pure HTML/CSS/JS, no
   build step.

## Project structure

```
.
├── main.py             # RAG pipeline (loading, chunking, embeddings, retrieval, chat)
├── server.py            # FastAPI wrapper exposing /upload and /chat
├── index.html            # Frontend UI
├── requirements.txt      # Python dependencies
└── README.md
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Add your API key

Create a file named `.env` in the project folder with:

```
GOOGLE_API_KEY=your_google_api_key_here
```

This is used for both the Gemini chat model and Google's embeddings. If it's
missing, the app falls back to local Hugging Face embeddings
(`sentence-transformers`), but you'll still need a working model for answering
questions.

## Running the app

You need **two things running at the same time**, in two separate terminals.

**Terminal 1 — start the backend API:**

```bash
uvicorn server:app --reload --port 8000
```

Wait for `Uvicorn running on http://127.0.0.1:8000`.

**Terminal 2 — serve the frontend:**

```bash
python -m http.server 5500
```

Then open **http://localhost:5500** in your browser (don't open `index.html`
directly as a file — it needs to be served for uploads to work reliably).

## Using it

1. Drag a PDF onto the upload area (or click to browse).
2. Wait for it to finish indexing — the catalog card will show page/chunk counts
   and stamp "Indexed" when ready.
3. Ask a question in the chat box. Answers include page-number citation chips
   pulled from the actual retrieved chunks.

If the backend isn't reachable, the interface still works — it clearly labels
answers as demo/placeholder responses so the UI stays usable without breaking.

## API reference (server.py)

| Endpoint | Method | Body | Response |
|---|---|---|---|
| `/upload` | POST | multipart `file` (PDF) | `{ session_id, pages, chunks, filename }` |
| `/chat` | POST | JSON `{ question, session_id }` | `{ answer, sources: [{ source, page }] }` |
| `/health` | GET | — | `{ status, active_sessions }` |
| `/session/{id}` | DELETE | — | `{ ended: session_id }` |

Each uploaded PDF gets its own Chroma vector store (session-scoped, kept in
memory), so multiple documents don't collide. Restarting the server clears all
sessions.

## Notes

- `allow_origins=["*"]` is set in `server.py`'s CORS config for local
  development. Restrict this to your actual frontend URL before deploying
  anywhere public.
- The API base URL the frontend calls is editable from the "Backend
  connection" panel in the sidebar (defaults to `http://localhost:8000`).
- Retriever strategy scales automatically with document size: basic similarity
  for small PDFs, similarity-with-threshold for medium ones, and multi-query
  retrieval for large ones.

## Troubleshooting

- **"Backend unreachable" in the UI** — check Terminal 1 is still running and
  didn't crash; check the API base URL in the settings panel matches the port
  `uvicorn` printed.
- **429 / quota errors** — you've hit your Google API rate limit; wait a
  minute and try again.
- **Import error on `from main import ...`** — `server.py` must sit in the
  same folder as `main.py`, and the file must be named exactly `main.py`.
