# Bajaj HackRx — RAG-Based Policy Question Answering System

A Retrieval-Augmented Generation (RAG) API built for the Bajaj HackRx challenge. The system accepts policy-document URLs and natural-language questions, retrieves relevant document passages using semantic search, and uses Google Gemini to generate answers grounded in the retrieved policy context.

## Overview

This project implements an end-to-end document intelligence pipeline:

1. Download a policy document from a URL.
2. Detect and parse supported document formats.
3. Clean and structurally chunk the extracted text.
4. Generate semantic embeddings with Sentence Transformers.
5. Build a FAISS vector index for similarity search.
6. Retrieve the most relevant chunks for each question.
7. Construct a grounded prompt containing the retrieved policy context.
8. Generate answers with Google Gemini.
9. Return answers through a FastAPI endpoint.
10. Cache processed documents so repeated requests do not rebuild the document index.

The repository also contains a local ETL/CLI workflow for preprocessing documents.

---

## Key Features

* **RAG-based question answering** over insurance and policy documents.
* **Multi-format document parsing** for PDF, DOCX, and email files.
* **Semantic text chunking** with section detection and sentence-aware splitting.
* **SentenceTransformer embeddings** using `all-MiniLM-L6-v2`.
* **FAISS vector search** using normalized embeddings and inner-product similarity.
* **Gemini-powered answer generation** using retrieved policy context.
* **Concurrent question processing** using `asyncio.gather`.
* **Document-level LRU caching** using `cachetools`.
* **Bearer-token authentication** for the HackRx endpoint.
* **FastAPI REST API** with interactive documentation.
* **Persistent FAISS artifacts** for the preprocessed document corpus.

---

## System Architecture

```text
                 ┌─────────────────────────┐
                 │       Client/User       │
                 └────────────┬────────────┘
                              │
                              │ POST /hackrx/run
                              ▼
                 ┌─────────────────────────┐
                 │       FastAPI API       │
                 │ Authentication + Request│
                 │       Validation        │
                 └────────────┬────────────┘
                              │
                 ┌────────────▼────────────┐
                 │   Document URL / Cache  │
                 │      LRUCache           │
                 └───────┬───────────┬─────┘
                         │           │
                   cache miss      cache hit
                         │           │
                         ▼           │
                ┌────────────────┐   │
                │ Download Docs  │   │
                └───────┬────────┘   │
                        ▼             │
                ┌────────────────┐   │
                │ Document Parser│   │
                │ PDF / DOCX /   │   │
                │ Email          │   │
                └───────┬────────┘   │
                        ▼             │
                ┌────────────────┐   │
                │ Text Chunking  │   │
                │ Sections + NLP │   │
                └───────┬────────┘   │
                        ▼             │
                ┌────────────────┐   │
                │ Embeddings     │   │
                │ MiniLM         │   │
                └───────┬────────┘   │
                        ▼             │
                ┌────────────────┐   │
                │ FAISS Index    │   │
                │ Similarity     │   │
                └───────┬────────┘   │
                        └──────┬──────┘
                               ▼
                    ┌──────────────────┐
                    │ Top-K Retrieval  │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ Grounded Prompt  │
                    │ + Policy Context │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ Google Gemini    │
                    │ Answer Generator │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ JSON Response    │
                    │ { "answers": [] }│
                    └──────────────────┘
```

---

## Tech Stack

| Layer           | Technology                                 |
| --------------- | ------------------------------------------ |
| API Framework   | FastAPI                                    |
| Language        | Python 3.11                                |
| Embedding Model | Sentence Transformers — `all-MiniLM-L6-v2` |
| Vector Search   | FAISS                                      |
| NLP / Chunking  | spaCy                                      |
| LLM             | Google Gemini API                          |
| PDF Parsing     | pdfminer.six                               |
| DOCX Parsing    | docx2txt                                   |
| Email Parsing   | mail-parser                                |
| MIME Detection  | python-magic                               |
| Caching         | cachetools LRUCache                        |
| HTTP            | requests                                   |

---

## Repository Structure

```text
BajajHackrx/
│
├── main.py
├── api_end.py
├── parsing.py
├── chunking.py
├── embeddings.py
├── faiss_indexing.py
│
├── requirements.txt
│
├── chunk_embeddings.npy
├── chunk_metadata.json
├── chunked_output.json
├── faiss_index.bin
├── parsed.txt
```

### File Responsibilities

### `main.py`

Main application entry point.

It contains:

* FastAPI application setup
* API authentication
* `/search` endpoint
* `/hackrx/run` endpoint
* document processing pipeline
* embedding generation
* FAISS retrieval
* Gemini answer generation
* LRU caching
* startup artifact loading
* ETL and CLI mode

### `api_end.py`

Provides the Google Gemini REST API integration.

The module:

* loads `GEMINI_API_KEY`
* creates the Gemini request payload
* sends the request using `requests`
* handles HTTP/API errors
* extracts generated text from the Gemini response

### `parsing.py`

Responsible for document detection and text extraction.

Supported formats:

* PDF
* DOCX
* Email

The parser uses MIME detection for byte input and file extensions for local file paths.

### `chunking.py`

Responsible for:

* cleaning extracted text
* detecting document sections
* sentence segmentation
* overlap-based chunk generation
* filtering boilerplate
* domain tagging
* splitting oversized chunks

### `embeddings.py`

Generates dense embeddings using:

```text
all-MiniLM-L6-v2
```

Embeddings are normalized and saved to:

```text
chunk_embeddings.npy
```

### `faiss_indexing.py`

Builds and persists the FAISS vector index.

The implementation uses:

```python
faiss.IndexFlatIP
```

with normalized vectors.

### `requirements.txt`

Contains the Python dependencies required by the application.

### `chunk_embeddings.npy`

Contains precomputed embedding vectors for the processed document chunks.

### `chunked_output.json`

Contains the generated chunks and associated metadata.

### `chunk_metadata.json`

Stores chunk metadata associated with the embedding pipeline.

### `faiss_index.bin`

Serialized FAISS index containing the document embeddings.

---

# RAG Pipeline

## 1. Document Ingestion

The `/hackrx/run` endpoint accepts a document URL through:

```json
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is covered by the policy?"
  ]
}
```

For a document that is not already cached, the API downloads the document using `requests`.

---

## 2. Document Parsing

The parser determines the document type based on its file extension or MIME type.

Supported document types:

```text
PDF
DOCX
Email
```

### PDF

PDF text is extracted using:

```text
pdfminer.six
```

### DOCX

DOCX text is extracted using:

```text
docx2txt
```

### Email

Email subject and body are extracted using:

```text
mail-parser
```

---

## 3. Text Cleaning

Before chunking, the text is cleaned.

The preprocessing removes content such as:

* page markers
* repeated blank lines
* carriage returns
* common boilerplate phrases
* very short irrelevant chunks

This reduces retrieval noise.

---

## 4. Section Detection

The chunking module attempts to identify structural sections using patterns such as:

```text
Section 1
Clause 2.3
Article 4
Chapter 5
Part 6
1.2
```

Each chunk retains the detected section title as metadata.

---

## 5. Semantic Chunking

The document is passed through spaCy sentence segmentation.

Chunks are created using sentence boundaries and a configurable overlap.

Long chunks are further split into smaller sentence-based pieces.

Each chunk stores metadata such as:

```json
{
  "doc_name": "...",
  "chunk_id": 1,
  "section_title": "...",
  "chunk_text": "...",
  "chunk_length": 120,
  "domain_tag": "coverage",
  "confidence": "high"
}
```

The chunking code also attempts to classify chunks into domains such as:

```text
coverage
exclusion
claims
eligibility
```

---

## 6. Embedding Generation

Each chunk is converted into a dense vector using:

```text
all-MiniLM-L6-v2
```

The embeddings are normalized before they are stored and indexed.

The embeddings are saved as:

```text
chunk_embeddings.npy
```

---

## 7. FAISS Indexing

The project uses:

```text
faiss.IndexFlatIP
```

for exact vector retrieval.

Because the embeddings are normalized, inner-product search provides cosine-similarity-style ranking.

The resulting index is saved as:

```text
faiss_index.bin
```

---

## 8. Query Embedding

For every user question, the same SentenceTransformer model converts the question into a normalized vector.

This puts the query and document chunks into the same embedding space.

---

## 9. Relevant Chunk Retrieval

FAISS searches the query vector against the indexed chunk vectors.

The current RAG answer-generation path retrieves:

```text
Top 3 chunks
```

for every question.

---

## 10. Prompt Construction

The retrieved chunks are combined into a policy context.

That context is inserted into a prompt instructing Gemini to:

* answer using the supplied policy documents
* remain concise and direct
* avoid relying on unsupported external information
* state when the requested information is not available

This is the retrieval-augmented generation step.

---

## 11. Gemini Answer Generation

The generated prompt is sent to Google's Gemini API.

The code uses:

```python
asyncio.to_thread(generate_answer_gemini, ...)
```

because the existing Gemini client implementation uses the blocking `requests` library.

---

## 12. Concurrent Question Processing

For a request containing multiple questions, the application creates a task for each question and executes:

```python
asyncio.gather(*tasks)
```

This enables multiple answer-generation operations to be scheduled concurrently.

---

## 13. Final Response

The API returns:

```json
{
  "answers": [
    "Answer to question 1",
    "Answer to question 2",
    "Answer to question 3"
  ]
}
```

---

# API Endpoints

## Root

```http
GET /
```

Returns a basic API welcome message.

---

## Semantic Search

```http
POST /search
```

Example request:

```json
{
  "query": "What is covered under the policy?",
  "top_k": 3
}
```

Example response:

```json
{
  "results": [
    {
      "score": 0.82,
      "chunk_id": 17,
      "section_title": "Section 4",
      "chunk_text": "..."
    }
  ]
}
```

This endpoint exposes the semantic retrieval layer independently of the full RAG answer-generation pipeline.

---

## HackRx RAG Endpoint

```http
POST /hackrx/run
```

Example request:

```json
{
  "documents": "https://example.com/policy.pdf",
  "questions": [
    "What is the coverage limit?",
    "What are the exclusions?"
  ]
}
```

Example response:

```json
{
  "answers": [
    "....",
    "...."
  ]
}
```

---

## Authentication Endpoint

```http
POST /api/v1/hackrx/token
```

Authentication is based on the environment variable:

```text
HACKRX_API_KEY
```

The endpoint returns a bearer token when the supplied credential is valid.

---

# Authentication Flow

The API uses FastAPI's OAuth2 password-form dependency.

The authentication process is:

```text
Client
   │
   │ username + password/API key
   ▼
/api/v1/hackrx/token
   │
   ├── validate against HACKRX_API_KEY
   │
   ▼
Bearer token
   │
   ▼
Authorization: Bearer <token>
   │
   ▼
/hackrx/run
```

---

# Caching

The application uses an in-memory LRU cache:

```python
LRUCache(maxsize=100)
```

The cache stores:

* processed chunks
* the temporary FAISS index

When the same document URL is requested again, the server can reuse the already processed representation instead of repeating:

```text
Download
   ↓
Parse
   ↓
Chunk
   ↓
Embed
   ↓
Build FAISS index
```

This can significantly reduce repeated document-processing overhead.

The cache is process-local, so separate application replicas would maintain separate caches.

---

# CLI / ETL Mode

Running:

```bash
python main.py
```

allows the application to operate in one of two modes.

## Mode 1 — ETL + Semantic Search CLI

The ETL workflow:

1. Loads `dataset/d1.pdf`
2. Parses the document
3. Chunks the extracted text
4. Writes `chunked_output.json`
5. Generates `chunk_embeddings.npy`
6. Builds `faiss_index.bin`
7. Loads the artifacts
8. Starts an interactive semantic-search CLI
9. Uses Gemini for CLI-generated answers

The local ETL code currently references:

```text
dataset/d1.pdf
```

which is not present in the repository tree currently inspected.

---

## Mode 2 — FastAPI Server

The server workflow:

1. Loads the SentenceTransformer model.
2. Loads the persisted FAISS index.
3. Loads the chunk metadata.
4. Starts the FastAPI application.

The application listens on:

```text
0.0.0.0:8000
```

---

# Local Setup

## Prerequisites

Install:

* Python 3.11
* pip
* Git
* a Google Gemini API key
* `libmagic` support where required by `python-magic`

---

## 1. Clone the Repository

```bash
git clone https://github.com/mightymehak/BajajHackrx.git
cd BajajHackrx
```

---

## 2. Create a Virtual Environment

### macOS / Linux

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Install the spaCy Model

The chunking pipeline uses the English spaCy model:

```bash
python -m spacy download en_core_web_sm
```

---

## 5. Configure Environment Variables

Create a local `.env` file:

```env
GEMINI_API_KEY=your_gemini_api_key
HACKRX_API_KEY=your_hackrx_api_key
```

Do not commit real credentials to the repository.

---

# Running the Project

Start the application:

```bash
python main.py
```

Choose:

```text
[1] Run ETL pipeline + semantic search CLI
[2] Serve FastAPI API
```

For API mode, the server runs on:

```text
http://localhost:8000
```

---

# Interactive API Documentation

FastAPI provides interactive API documentation.

Once the application is running, access:

```text
/docs
```

or:

```text
/redoc
```

These interfaces can be used to inspect and test the available endpoints.

---

# Performance-Oriented Design

## Normalized Embeddings

The project normalizes document and query embeddings before retrieval.

Combined with:

```python
faiss.IndexFlatIP
```

this provides cosine-similarity-style ranking.

---

## Precomputed Index

The initial document corpus can be processed offline.

The resulting artifacts:

```text
chunk_embeddings.npy
faiss_index.bin
chunk_metadata.json
```

can then be loaded directly by the FastAPI server.

This avoids recomputing embeddings on every startup.

---

## Concurrent LLM Calls

Multiple questions are processed using:

```python
asyncio.gather(...)
```

while the blocking Gemini request is moved to a worker thread with:

```python
asyncio.to_thread(...)
```

This allows the API to handle a multi-question request more efficiently than serially waiting for each Gemini call.

---

## Document-Level Caching

The application caches processed documents by URL.

This avoids repeated parsing and embedding generation for the same document during the lifetime of the application process.

---

# Limitations

The current implementation has a few practical limitations:

* The LRU cache is in-memory and process-local.
* `IndexFlatIP` performs exact vector search and can become more expensive for very large vector collections.
* The SentenceTransformer model remains loaded in application memory.
* Gemini generation depends on external network/API availability.
* The RAG generation path currently retrieves a fixed top-3 set of chunks.
* The persistent vector artifacts correspond to the corpus used during preprocessing.
* The local ETL flow references `dataset/d1.pdf`, which is not currently included in the repository.
* The Gemini client currently uses synchronous `requests`, with concurrency achieved through `asyncio.to_thread`.

---

# Future Improvements

Potential extensions include:

* HNSW or IVF FAISS indexes for larger datasets.
* Hybrid lexical + semantic retrieval.
* Cross-encoder reranking after vector retrieval.
* Better document hierarchy and table extraction.
* Redis-based distributed caching.
* Request rate limiting.
* Structured logging and observability.
* Automated unit and integration tests.
* Source citations returned with every answer.
* Fully asynchronous HTTP calls using `httpx`.
* Document versioning and cache invalidation.
* Stronger prompt injection and untrusted-document handling.

---

# End-to-End Flow

```text
Client
  │
  │ Document URL + Questions
  ▼
FastAPI
  │
  ├── Authenticate request
  │
  ├── Check document cache
  │
  ├── Download document if needed
  │
  ├── Detect document type
  │
  ├── Parse document
  │
  ├── Clean text
  │
  ├── Detect sections
  │
  ├── Create semantic chunks
  │
  ├── Generate embeddings
  │
  ├── Build/search FAISS
  │
  ├── Retrieve top relevant chunks
  │
  ├── Build grounded prompt
  │
  ├── Call Gemini
  │
  └── Return JSON answers
  ▼
Client
```

---

# Why This Project Matters

This project demonstrates a complete RAG workflow instead of simply calling an LLM.

It combines:

* document ingestion
* document parsing
* NLP preprocessing
* semantic chunking
* dense embeddings
* vector search
* retrieval-augmented prompting
* LLM inference
* API authentication
* caching
* asynchronous request handling

The result is an end-to-end system designed to answer natural-language questions over unstructured policy documents.

---

# Author

**Mehakpreet Kaur**

GitHub: `@mightymehak`

---

# License

No explicit open-source license file is currently present in the repository. Add a license before distributing the project under an open-source license.
