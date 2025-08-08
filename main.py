import os
import json
import shutil
import requests
from pathlib import Path
from io import BytesIO
from typing import Optional, List, Dict
import asyncio
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
load_dotenv()

# Import your custom modules
from parsing import parse_document
from chunking import chunk_document
from embeddings import generate_and_save_embeddings
from faiss_indexing import build_faiss_index

# Import the refactored Gemini API function.
# We will use an asynchronous version for concurrency.
from api_end import generate_answer_gemini

# Import FastAPI libraries, including security modules
from fastapi import FastAPI, HTTPException, Response, Depends, status
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, HttpUrl

# ---- New Caching Import ----
from cachetools import LRUCache


# ---- Configurable paths ---- #
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = "faiss_index.bin"
CHUNK_METADATA_PATH = "chunked_output.json"
DEFAULT_TOP_K = 3

# --- New Global for managing uploaded documents ---
UPLOAD_DIR = Path("uploaded_documents")
UPLOAD_DIR.mkdir(exist_ok=True)

# ---- Authentication Configuration ----
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# ---- FastAPI app ---- #
app = FastAPI(
    title="Bajaj Hackrx API",
    description="LLM powered semantic search and question answering API for the Bajaj Hackrx.",
    version="4.0",
    docs_url="/api/v1/hackrx/run"
)

@app.on_event("startup")
async def startup_event():
    print("Loading embedding model and initial FAISS index during startup...")
    load_artifacts()

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)

@app.get("/.well-known/appspecific/com.chrome.devtools.json", include_in_schema=False)
async def chrome_devtools_json():
    return Response(status_code=204)

# ---- Pydantic Models ----
class QueryRequest(BaseModel):
    query: str
    top_k: int = DEFAULT_TOP_K

class HackathonRequest(BaseModel):
    documents: HttpUrl
    questions: list[str]

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

# ---- Globals for embedding model, faiss, and metadata ----
model = None
faiss_index = None
chunk_metadata = None

# ---- New Caching Object ----
processed_doc_cache = LRUCache(maxsize=100)


def load_artifacts():
    global model, faiss_index, chunk_metadata
    print("Loading embedding model and FAISS index for search/API...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    
    if Path(FAISS_INDEX_PATH).exists() and Path(CHUNK_METADATA_PATH).exists():
        faiss_index = faiss.read_index(FAISS_INDEX_PATH)
        with open(CHUNK_METADATA_PATH, "r", encoding="utf-8") as f:
            chunk_metadata = json.load(f)
        print("Initial artifacts loaded. Ready for API.")
    else:
        print("Warning: Initial artifacts not found. Please run ETL pipeline first.")
        faiss_index = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())
        chunk_metadata = []

# ---- New Authentication Endpoint ----
HACKRX_API_KEY = os.getenv("HACKRX_API_KEY")
@app.post("/api/v1/hackrx/token", tags=["Authentication"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()) -> TokenResponse:
    if form_data.password != HACKRX_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"access_token": HACKRX_API_KEY, "token_type": "bearer"}


# --- New async helper function for concurrent processing ---
async def process_single_question_async(query: str, temp_faiss_index: faiss.Index, chunks: List[Dict], model) -> str:
    """
    Handles the RAG process for a single question asynchronously.
    """
    try:
        query_emb = model.encode([query], normalize_embeddings=True).astype('float32')
        distances, indices = temp_faiss_index.search(query_emb, k=3)
        
        retrieved_chunks = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(chunks):
                continue
            retrieved_chunks.append(chunks[idx])
        
        context_text = "\n\n".join(chunk.get("chunk_text", "") for chunk in retrieved_chunks)

        simple_prompt = f"""
        Answer the following question based *only* on the provided policy documents. Be concise and direct. If the information is not available in the documents, state "Information not found in the provided policy."
        
        Policy Documents:
        {context_text}
        
        Question:
        {query}
        
        Answer:
        """
        
        # Use asyncio.to_thread to run the blocking Gemini API call in a thread pool.
        # This is a safe way to handle blocking I/O calls in an async function.
        answer_text = await asyncio.to_thread(generate_answer_gemini, simple_prompt, max_tokens=256)
        
        cleaned_answer = answer_text.strip()
        if cleaned_answer.startswith("Answer:"):
            cleaned_answer = cleaned_answer[len("Answer:"):].strip()
        
        return cleaned_answer
        
    except Exception as e:
        print(f"Error generating answer for query '{query}': {e}")
        return "An error occurred while generating the answer."


# ---- FastAPI endpoints ----
@app.get("/")
def root():
    return {
        "message": "Welcome to the Semantic Search & QA API! Use /docs for API documentation."
    }

@app.post("/search")
def semantic_search_endpoint(req: QueryRequest):
    query = req.query.strip()
    k = req.top_k if req.top_k else DEFAULT_TOP_K
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if model is None or faiss_index is None or chunk_metadata is None:
        raise HTTPException(status_code=503, detail="System not ready. Please run ETL pipeline.")
    query_emb = model.encode([query], normalize_embeddings=True).astype('float32')
    distances, indices = faiss_index.search(query_emb, k)
    results = []
    for score, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(chunk_metadata):
            continue
        chunk = chunk_metadata[idx]
        results.append({
            "score": float(score),
            "chunk_id": chunk.get("chunk_id"),
            "section_title": chunk.get("section_title"),
            "chunk_text": chunk.get("chunk_text", "")[:300] + ("..." if len(chunk.get("chunk_text", "")) > 300 else "")
        })
    if not results:
        return {"results": [{"chunk_text": "No relevant information found."}]}
    return {"results": results}


@app.post("/hackrx/run")
async def hackrx_run_endpoint(req: HackathonRequest, token: str = Depends(oauth2_scheme)):
    print(f"Hackathon request received.")
    if model is None:
        raise HTTPException(status_code=503, detail="Embedding model not loaded.")
    
    document_url = str(req.documents)
    
    # ---- START CACHING LOGIC ----
    # Check if the processed document is already in our cache
    cached_data = processed_doc_cache.get(document_url)
    
    if cached_data:
        print(f"Using cached data for document: {document_url}")
        temp_faiss_index = cached_data['faiss_index']
        chunks = cached_data['chunks']
    else:
        # If not in cache, proceed with the full processing pipeline
        print(f"Processing document from scratch: {document_url}")
        
        temp_faiss_index = None
        chunks = []

        try:
            print(f"Downloading document from: {req.documents}")
            response = await asyncio.to_thread(requests.get, document_url)
            response.raise_for_status()
            
            print("Processing document and building temporary index...")
            document_bytes = response.content
            parsed_text = parse_document(document_bytes)
            
            if not parsed_text:
                raise Exception("Parsing failed for the provided document.")
                
            chunks = chunk_document(parsed_text, doc_name=document_url, overlap=1)
            if not chunks:
                raise Exception("Chunking failed for the provided document.")

            new_chunk_texts = [chunk['chunk_text'] for chunk in chunks]
            new_embeddings = model.encode(new_chunk_texts, normalize_embeddings=True).astype('float32')
            
            temp_faiss_index = faiss.IndexFlatL2(model.get_sentence_embedding_dimension())
            temp_faiss_index.add(new_embeddings)

            # Store the processed data in the cache for future requests
            processed_doc_cache[document_url] = {
                'faiss_index': temp_faiss_index,
                'chunks': chunks
            }
            
        except Exception as e:
            print(f"Error during document processing for /hackrx/run: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to process the document from URL. Error: {e}"
            )
    # ---- END CACHING LOGIC ----
    
    # --- The concurrent answer generation part ---
    tasks = [
        process_single_question_async(query, temp_faiss_index, chunks, model)
        for query in req.questions
    ]
    
    try:
        answers_list = await asyncio.gather(*tasks)
    except Exception as e:
        print(f"An error occurred during concurrent answer generation: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate one or more answers.")
    
    return {"answers": answers_list}

# ---- ETL Pipeline & CLI Search ---- #
def etl_and_cli():
    file_path = "dataset/d1.pdf"
    UPLOAD_DIR.mkdir(exist_ok=True)
    print(f"Parsing document: {file_path}")
    parsed_text = parse_document(file_path)
    if not parsed_text:
        print("Parsing failed. Exiting.")
        exit(1)
    chunks = chunk_document(parsed_text, doc_name=file_path, overlap=1)
    with open("chunked_output.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)
    print(f"Chunking complete: Generated {len(chunks)} chunks and saved to chunked_output.json\n")
    for c in chunks[:5]:
        print(f"Chunk ID: {c['chunk_id']} | Section: {c['section_title']} | Domain: {c['domain_tag']} | Confidence: {c['confidence']}")
        print(f"Text (first 150 chars): {c['chunk_text'][:150]}...\n{'='*60}\n")
    generate_and_save_embeddings(chunks)
    print("Building FAISS index on generated embeddings...")
    build_faiss_index(
        embedding_path='chunk_embeddings.npy',
        metadata_path='chunked_output.json',
        index_path='faiss_index.bin'
    )
    print("FAISS vector database setup complete.")
    print("\n=== Semantic Search CLI ===")
    load_artifacts()
    while True:
        query = input("\nEnter your search query (or type 'quit' to exit): ").strip()
        if query.lower() == "quit":
            print("Exiting search.")
            break
        query_emb = model.encode([query], normalize_embeddings=True).astype('float32')
        distances, indices = faiss_index.search(query_emb, DEFAULT_TOP_K)
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(chunk_metadata):
                continue
            chunk = chunk_metadata[idx]
            results.append({
                "score": float(score),
                "section_title": chunk.get("section_title"),
                "text": chunk.get("chunk_text", ""),
            })
        print("\nTop matches:")
        for i, r in enumerate(results, 1):
            print(f"\nResult {i}: (score={r['score']:.4f})")
            print(f"Section: {r['section_title']}")
            print(f"Text: {r['text'][:300]}...")
            print("="*100)
        context = '\n\n'.join([r['text'] for r in results])
        prompt = f"""
You are an expert claims processor. Your task is to analyze a user's query based on the provided insurance policy documents and return a structured JSON response.
**Instructions:**
... (structured prompt remains the same) ...
**User Query:**
{query}
**Context (Policy Clauses):**
{context}
**Desired JSON Output Format:**
{{
    "Decision": "[Approved|Rejected|Information|Requires More Information]",
    "Amount": "[<numeric value> | \"N/A\"]",
    "Justification": "..."
}}
**Your JSON response:**
"""
        print("\nGemini Answer API Response:")
        try:
            raw_llm_response = generate_answer_gemini(prompt, max_tokens=512)
            clean_response = raw_llm_response.strip().strip('```json').strip('`')
            final_answer = json.loads(clean_response)
            print(json.dumps(final_answer, indent=2))
        except Exception as e:
            print(f"Gemini API call failed: {e}")

# ---- Entrypoint ---- #
if __name__ == "__main__":
    print("Select mode:")
    print("[1] Run ETL pipeline + semantic search CLI (Initial setup or batch processing)")
    print("[2] Serve FastAPI API (/search, /hackrx/run)")
    try:
        user_mode = input("Enter 1 or 2: ").strip()
    except Exception:
        user_mode = None
    if user_mode == "1":
        etl_and_cli()
    elif user_mode == "2":
        load_artifacts()
        import uvicorn
        print("Starting FastAPI app on [http://0.0.0.0:8000](http://0.0.0.0:8000)")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        print("Invalid input. Exiting.")