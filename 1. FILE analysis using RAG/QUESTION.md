#

**Problem:**
A small company ("EjobIndia," going by your files) wants an internal tool where employees can ask questions about a **50+ page policy/handbook PDF** (or any PDF you pick — even a large one, like a full textbook chapter). Your current approach (`gpt.py`, `_stream.py` etc.) would either blow past the context window, get slow, or get expensive since you resend the whole document on every question. They've asked you to fix it.

**Your role:**
You're the backend/AI engineer responsible for making this Q&A tool actually scale.

**Goal:**
Replace full-text-stuffing with a real retrieval step: chunk the PDF, embed the chunks, store them in a local vector store, and at query time retrieve only the top-k relevant chunks to feed the LLM — instead of the whole PDF.

**Requirements:**

- Load a PDF (reuse your `pypdf` knowledge, or `PyPDFLoader` if you want to try LangChain's loader).
- Split into chunks (~400–500 tokens, ~10–20% overlap). You can do this manually with a simple sliding-window splitter, or use `RecursiveCharacterTextSplitter` — try writing your own splitter first, it'll teach you more than importing one.
- Generate embeddings for each chunk (OpenAI `text-embedding-3-small` or a Gemini embedding model — you already have API keys wired up).
- Store embeddings + chunk text (+ page number metadata) in a vector store — start with **FAISS** or **Chroma** since both are simple, local, no server needed.
- At query time: embed the user's question, retrieve top-3–5 most similar chunks, build a prompt from *just those chunks*, and call the LLM.
- Print (or Streamlit-display) which chunks were retrieved, not just the final answer — you need to see what's actually being fed to the model to debug retrieval quality.

**Constraints:**

- No cloud-hosted vector DB (Pinecone etc.) — keep it local/free for this session.
- Must handle a PDF too large to fit in one prompt (test this — pick something genuinely big, 50+ pages, not your small "EjobIndia.pdf").
- Reuse your existing `.env` / dotenv pattern for API keys.

**Expected output:**
A single script (or Streamlit app, your choice) that: ingests a PDF once (build the index), then lets you ask multiple questions against it, each time retrieving and printing the relevant chunks before answering.

**Success criteria:**

- Ask a question whose answer is on page 40 of a 60-page doc — the system should retrieve a chunk containing that answer, not the whole document.
- Ask something *not* in the PDF at all — a good retrieval system should either say "not found" or retrieve irrelevant chunks (you should notice this and see why grounding-check prompts still matter).
- Compare token usage/prompt size against your old full-stuffing approach — you should be able to point at the difference concretely.

**What you're allowed to use:**

- Already know: PdfReader/PyPDF, OpenAI/Gemini SDK calls, `.env` handling, basic prompt construction, Streamlit basics.
- Consult docs freely: FAISS/Chroma docs, OpenAI/Gemini embeddings API docs, LangChain text-splitter docs if you use it.
- New concepts you'll likely need to learn as you go: what an embedding vector actually is/does, cosine similarity search, how a vector index stores/queries vectors — don't look up "how to build RAG" tutorials wholesale, but do look up specific API references when stuck.

Bring back your attempt (or wherever you get stuck) and I'll review it like a mentor — not rewrite it for you.
