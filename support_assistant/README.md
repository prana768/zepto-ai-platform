# Zepto Support Assistant

## Overview

The Zepto Support Assistant answers questions about Zepto policies using a local document corpus, local embeddings, ChromaDB retrieval, LangGraph orchestration, structured Pydantic output, and a FastAPI API.

The baseline runs deterministically with MOCK_LLM=1.

## Architecture

Documents
↓
Chunking
↓
Local Embeddings
↓
ChromaDB
↓
User Query
↓
LangGraph Intent Classification
↓
Policy Retrieval OR Direct Answer
↓
Structured Pydantic Response
↓
FastAPI /ask

## Components

| Stage | Component | File / Node |
|---|---|---|
| Document ingestion | Policy documents | docs/doc_01.txt - docs/doc_08.txt |
| Embedding | Sentence Transformers | all-MiniLM-L6-v2 |
| Vector database | ChromaDB | chroma_db/ |
| Orchestration | LangGraph | app.py |
| Intent classification | LangGraph node | classify_intent |
| Retrieval + answer | LangGraph node | retrieve_and_answer |
| Direct response | LangGraph node | direct_answer |
| Structured output | Pydantic | SupportResponse |
| API | FastAPI | POST /ask |
| Containerization | Docker | Dockerfile |

## Document Corpus

The system contains eight Zepto policy documents:

1. Delivery Policy
2. Returns & Refunds
3. Membership
4. Tracking
5. Cancellation
6. Damaged/Missing
7. Gift Cards
8. Support Hours

## Embedding and Retrieval

The local embedding model is `all-MiniLM-L6-v2`.

The documents are embedded and stored in ChromaDB using cosine similarity.

For policy questions, the system retrieves the top 3 most similar documents.

The deterministic response uses the most similar retrieved document as the grounded context.

## LangGraph Workflow

### 1. classify_intent

The query is converted to lowercase and checked for policy-related keywords:

- delivery
- return
- refund
- membership
- tracking
- cancel
- gift card
- support hours

Matching queries are classified as `policy_question`.

Other queries are classified as `general_question`.

### 2. retrieve_and_answer

Policy questions are embedded and matched against the ChromaDB collection.

The top 3 similar documents are retrieved.

The deterministic response is generated from the most similar retrieved context.

### 3. direct_answer

General questions receive:

> I can only answer questions about Zepto policies right now.

## Structured Output

Every API response follows this schema:

```json
{
  "answer": "string",
  "sources": ["document_id"],
  "confidence": 1.0
}
```

The confidence value is constrained between 0 and 1.

Policy questions return retrieved document IDs in `sources`.

General questions return an empty `sources` list.

## MOCK_LLM Mode

The baseline uses `MOCK_LLM=1`.

This mode:

- requires no API key
- provides deterministic intent classification
- uses deterministic retrieval
- returns deterministic responses

The routing decision does not depend on an LLM.

An optional real-LLM path can be enabled using `MOCK_LLM=0`.

## FastAPI

The service exposes:

`POST /ask`

### Example policy request

```json
{
  "query": "What is the delivery charge?"
}
```

### Example policy response

```json
{
  "answer": "Based on the retrieved context: Delivery Policy ...",
  "sources": ["doc_01", "doc_03", "doc_04"],
  "confidence": 1.0
}
```

### Example general request

```json
{
  "query": "What is the capital of India?"
}
```

### Example general response

```json
{
  "answer": "I can only answer questions about Zepto policies right now.",
  "sources": [],
  "confidence": 1.0
}
```

## Running Locally

Install dependencies:

```bash
pip install fastapi uvicorn pydantic langgraph chromadb sentence-transformers
```

Set deterministic mock mode:

```bash
export MOCK_LLM=1
```

Run:

```bash
uvicorn app:app --host 0.0.0.0 --port 7860
```

## Docker

Build:

```bash
docker build -t zepto-support-assistant .
```

Run:

```bash
docker run -p 7860:7860 zepto-support-assistant
```

## Project Structure

```text
support_assistant/
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
├── chroma_db/
├── app.py
├── Dockerfile
└── README.md
```