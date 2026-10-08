# MyManah AI — Journal Analysis & Document Intelligence

An AI-powered backend system for journal analysis and document-based question answering using locally hosted open-source Hugging Face models.

The system combines:
- Sentiment analysis
- Emotion detection
- Mood scoring
- Crisis-risk screening
- Journal summarization
- Retrieval-Augmented Generation (RAG)
- PDF document ingestion
- Semantic search using embeddings
- FAISS vector search
- Local LLM inference
- FastAPI REST APIs

The AI inference pipeline is designed to run locally without proprietary hosted LLM APIs.

---

## Overview

MyManah AI is a modular AI backend with two major capabilities.

### Journal Analysis

A journal entry can be submitted through a REST API and analyzed using locally downloaded transformer models.

The system extracts:
- Sentiment
- Dominant emotion
- Mood score
- Short summary
- Crisis risk
- Confidence score

### Document Question Answering

The RAG pipeline allows users to upload a PDF and ask questions about its content.

Pipeline:

```text
PDF
 ↓
Text Extraction
 ↓
Text Chunking
 ↓
Embedding Generation
 ↓
FAISS Vector Index
 ↓
Semantic Retrieval
 ↓
Retrieved Context
 ↓
Local Qwen LLM
 ↓
Grounded Answer + Sources
```

---

## Key Features

### Journal Intelligence
- Local transformer-based sentiment analysis
- Multi-class emotion classification
- Mood score generation
- Crisis-risk screening
- Context-aware crisis decision layer
- Journal summarization
- Structured JSON responses

### Document Intelligence
- PDF upload
- Page-aware text extraction
- Text chunking
- Semantic embeddings
- FAISS vector search
- Top-K retrieval
- Local LLM generation
- Source page and chunk references
- Document-grounded answers
- Fallback for unsupported questions

### Engineering
- FastAPI REST API
- Pydantic validation
- Modular service architecture
- Separation of API, business logic, and model layers
- Local inference
- Swagger/OpenAPI documentation
- Component-level testing

---

## System Architecture

```text
                         MyManah AI
                              |
              +---------------+---------------+
              |                               |
              v                               v
       JOURNAL ANALYSIS                 DOCUMENT RAG
              |                               |
       +------+------+                 PDF Upload
       |      |      |                      |
       v      v      v                      v
   Sentiment Emotion Crisis             PDF Loader
       |      |      |                      |
       +------+------+                      v
              |                       Text Splitter
              v                             |
        Mood / Summary                      v
              |                        Embeddings
              v                             |
       Structured Result                    v
                                          FAISS
                                            |
                                      User Question
                                            |
                                            v
                                      Top-K Retrieval
                                            |
                                            v
                                    Retrieved Context
                                            |
                                            v
                                   Local Qwen LLM
                                            |
                                            v
                                      Answer + Sources
```

---

# Journal Analysis

Example request:

```json
{
  "text": "Today was an amazing day. I spent time with my family and I feel extremely happy."
}
```

Example response:

```json
{
  "sentiment": "positive",
  "emotion": "happy",
  "moodScore": 8,
  "summary": "The writer describes a positive day spent with family and expresses strong happiness.",
  "crisisRisk": "LOW",
  "confidence": 0.98
}
```

Exact confidence and classification values depend on model output.

### Journal Pipeline

```text
Journal Text
     |
     +----------------------+
     |                      |
     v                      v
Sentiment Model        Emotion Model
     |                      |
     +----------+-----------+
                |
                v
          Mood Scoring
                |
                v
        Summary Generation
                |
                v
         Crisis Detection
                |
                v
        Context Decision Layer
                |
                v
        Structured API Response
```

---

# Sentiment Analysis

Model:

```text
cardiffnlp/twitter-roberta-base-sentiment-latest
```

The model provides sentiment probabilities which are converted into the structured application response.

It is downloaded and executed locally using Hugging Face Transformers.

The model was selected because it is open-source, supports local inference, provides direct sentiment classification, and integrates with the Transformers ecosystem.

### Limitation

The model was trained on social-media-style text rather than specifically on private journal entries. Domain-specific validation would improve production reliability.

---

# Emotion Detection

Model:

```text
SamLowe/roberta-base-go_emotions
```

The model is based on RoBERTa and trained on the GoEmotions dataset.

The application maps model emotions into these application-level categories:

- Happy
- Sad
- Anxiety
- Stress
- Anger
- Fear
- Neutral

The mapping layer keeps the underlying model independent from the application's output vocabulary.

---

# Mood Score

The mood score is represented on a scale of:

```text
1 - 10
```

The application-level score uses signals from journal analysis such as:

- Sentiment polarity
- Emotion category
- Model confidence
- Emotional intensity

The score is an application-level interpretation and is not a clinically validated psychological measurement.

---

# Crisis Risk Detection

Crisis detection uses a two-stage architecture.

```text
Journal Text
     |
     v
ModernBERT Crisis Classifier
     |
     v
Crisis Probability
     |
     v
Context-Aware Decision Layer
     |
     +--------+---------+
     |        |         |
     v        v         v
    LOW    MEDIUM      HIGH
```

Model:

```text
Akashpaul123/modernbert-crisis-detection
```

The classifier provides a crisis/non-crisis signal. A lightweight context layer considers:

- Explicit denial
- Third-person references
- Historical references
- Suicidal ideation
- Explicit planning
- Imminent self-harm language

The raw model probability is therefore not treated as a definitive risk decision.

### Safety Limitation

The crisis component is a technical screening mechanism, not a medical diagnosis, clinical assessment, professional mental-health evaluation, emergency triage system, or definitive assessment of a person's safety.

A production system would require domain-specific labelled data, independent validation, calibration, false-positive/false-negative analysis, subgroup evaluation, human review, and appropriate clinical/safety validation.

---

# Journal Summarization

The application includes a local summarization component that produces a concise summary of the journal entry.

The summarization model runs locally and does not depend on a hosted LLM API.

---

# RAG Document Intelligence

The RAG system allows users to upload a PDF and ask questions about its content.

```text
PDF
 |
 v
Text Extraction
 |
 v
Text Chunking
 |
 v
Embedding Generation
 |
 v
FAISS Vector Index
 |
 v
Question
 |
 v
Semantic Retrieval
 |
 v
Top-K Relevant Chunks
 |
 v
Context Construction
 |
 v
Local Qwen LLM
 |
 v
Grounded Answer
```

---

# PDF Processing

The PDF loader extracts text while preserving page information.

Each extracted page retains its page number so retrieved content can later be traced back to the original document.

---

# Text Chunking

Long documents are divided into smaller chunks before embedding.

Each chunk maintains metadata such as:

```text
Page Number
Chunk ID
Text Content
```

Chunking allows semantic search to operate on focused portions of the document rather than the entire document.

---

# Embeddings

Each document chunk is converted into a numerical vector using a local Sentence Transformers embedding model.

The user question is embedded using the same embedding model.

The system then searches for document chunks that are semantically similar to the query.

---

# FAISS Vector Search

FAISS is used for local vector similarity search.

```text
Document Chunks
      |
      v
Embeddings
      |
      v
FAISS Index
      |
User Query
      |
      v
Query Embedding
      |
      v
Similarity Search
      |
      v
Top-K Chunks
```

---

# Retrieval

The retriever returns the most relevant chunks.

Example:

```json
{
  "page": 2,
  "chunk": 1,
  "similarity": 0.6449
}
```

Similarity scores are retained for transparency and debugging.

---

# Local RAG Generation

The generation model is:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

The model is downloaded locally and executed with PyTorch and Hugging Face Transformers.

It receives the user question and retrieved document context.

---

# Grounded Generation

The RAG generator is instructed to:

1. Answer only from the supplied document context.
2. Avoid unrelated pretrained knowledge.
3. Avoid unsupported guesses.
4. Return a fallback response when the answer is not present.
5. Keep the answer concise and factual.

Fallback response:

```text
The answer is not available in the provided document.
```

---

# Source Attribution

RAG responses include retrieved source information.

Example:

```json
{
  "answer": "Deep learning is a subset of machine learning...",
  "sources": [
    {
      "page": 2,
      "chunk": 1,
      "similarity": 0.6449
    },
    {
      "page": 7,
      "chunk": 14,
      "similarity": 0.6439
    }
  ]
}
```

This makes retrieved evidence traceable to the original document.

---

# Project Structure

```text
mymanah-ai/
│
├── app/
│   ├── api/
│   │   └── routes/
│   │       ├── journal.py
│   │       └── rag.py
│   │
│   ├── models/
│   │   ├── sentiment.py
│   │   ├── emotion.py
│   │   ├── crisis.py
│   │   ├── mood.py
│   │   └── summarizer.py
│   │
│   ├── services/
│   │   ├── journal_service.py
│   │   ├── emotion_service.py
│   │   ├── crisis_service.py
│   │   └── ...
│   │
│   ├── rag/
│   │   ├── loader.py
│   │   ├── splitter.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   └── generator.py
│   │
│   ├── schemas/
│   │   ├── journal.py
│   │   └── rag.py
│   │
│   ├── main.py
│   └── __init__.py
│
├── data/
├── tests/
├── .gitignore
├── README.md
├── requirements.txt
└── ...
```

---

# Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| Backend | FastAPI |
| Server | Uvicorn |
| Validation | Pydantic |
| ML Framework | PyTorch |
| NLP | Hugging Face Transformers |
| Embeddings | Sentence Transformers |
| Vector Search | FAISS |
| Sentiment | RoBERTa |
| Emotion | RoBERTa / GoEmotions |
| Crisis | ModernBERT |
| RAG Generation | Qwen2.5-1.5B-Instruct |
| API Documentation | OpenAPI / Swagger |

---

# Model Selection

The project uses specialized models instead of forcing one model to perform every task.

| Task | Model / Technology | Purpose |
|---|---|---|
| Sentiment | CardiffNLP RoBERTa | Sentiment classification |
| Emotion | RoBERTa GoEmotions | Emotion classification |
| Crisis | ModernBERT crisis classifier | Crisis screening signal |
| Embeddings | Sentence Transformers | Semantic representation |
| Vector Search | FAISS | Similarity search |
| Generation | Qwen2.5-1.5B-Instruct | Local grounded generation |
| API | FastAPI | REST API layer |

---

# Why Local Models?

The system is designed around local inference.

Benefits include:

- No dependency on proprietary LLM APIs
- Greater control over model execution
- No per-request external inference cost
- Sensitive data can remain on the local system
- Easier experimentation with open-source models
- Clear separation between application and model providers

The trade-off is higher local CPU/RAM/GPU requirements.

---

# Installation

## Requirements

Recommended environment:

```text
Python 3.12
```

### Clone

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd mymanah-ai
```

### Create Virtual Environment

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Application

```bash
uvicorn app.main:app --reload
```

Application:

```text
http://127.0.0.1:8000
```

Swagger UI:

```text
http://127.0.0.1:8000/docs
```

OpenAPI schema:

```text
http://127.0.0.1:8000/openapi.json
```

Models are downloaded from Hugging Face on first initialization and can then be reused from the local cache.

---

# API Documentation

FastAPI provides interactive Swagger documentation at:

```text
http://127.0.0.1:8000/docs
```

It can be used to inspect schemas, upload PDFs, and execute API requests.

---

# Journal Analysis API

## Endpoint

```http
POST /analyze-journal
```

## Request

```json
{
  "text": "Today was an amazing day. I spent time with my family and I feel extremely happy."
}
```

## Response

```json
{
  "sentiment": "positive",
  "emotion": "happy",
  "moodScore": 8,
  "summary": "The journal describes a positive day spent with family and strong feelings of happiness.",
  "crisisRisk": "LOW",
  "confidence": 0.98
}
```

---

# RAG APIs

The RAG functionality provides:

```text
POST /rag/upload
POST /rag/query
```

## Upload PDF

```http
POST /rag/upload
```

The endpoint accepts a PDF using multipart form-data.

Example response:

```json
{
  "message": "PDF uploaded and indexed successfully.",
  "pages": 512,
  "chunks": 2841
}
```

## Query Document

```http
POST /rag/query
```

Request:

```json
{
  "question": "What is deep learning?"
}
```

Example response:

```json
{
  "answer": "Deep learning is a subset of machine learning where artificial neural networks with multiple layers are used for pattern recognition tasks.",
  "sources": [
    {
      "page": 2,
      "chunk": 1,
      "similarity": 0.6449
    },
    {
      "page": 7,
      "chunk": 14,
      "similarity": 0.6439
    },
    {
      "page": 474,
      "chunk": 2676,
      "similarity": 0.6047
    }
  ]
}
```

---

# Document Grounding Example

For a question unrelated to the uploaded document:

```json
{
  "question": "What is the capital of France?"
}
```

the system is designed to return:

```json
{
  "answer": "The answer is not available in the provided document.",
  "sources": []
}
```

---

# Example Workflow

```text
1. Start FastAPI
        |
        v
2. Open Swagger
        |
        v
3. Upload PDF
        |
        v
4. Extract text
        |
        v
5. Create chunks
        |
        v
6. Generate embeddings
        |
        v
7. Build FAISS index
        |
        v
8. Submit question
        |
        v
9. Retrieve top-K chunks
        |
        v
10. Generate grounded answer
        |
        v
11. Return answer + sources
```

---

# Design Decisions

### Modular AI Components

Sentiment, emotion, crisis, retrieval, and generation components are separated so models can be replaced independently.

### Service Layer

Business logic is kept outside FastAPI routes:

```text
API Route
    ↓
Service
    ↓
Model / RAG Component
```

### Local Inference

Models run locally instead of relying on remote inference APIs.

### Specialized Models

Different models are used for different tasks rather than using a single large model for everything.

### Retrieval Before Generation

The RAG pipeline retrieves relevant document chunks before invoking the language model.

### Source Metadata

Page numbers, chunk identifiers, and similarity scores are preserved for traceability.

---

# Testing

Component-level and integration-oriented tests are included.

```bash
python -m tests.test_sentiment
python -m tests.test_emotion
python -m tests.test_crisis
python -m tests.test_crisis_service
python -m tests.test_rag_loader
python -m tests.test_rag
python -m tests.test_rag_generation
```

The API can also be tested interactively through Swagger.

---

# Performance Considerations

Potential performance bottlenecks include:

- Initial model loading
- PDF processing
- Embedding generation
- Large document indexing
- Local LLM generation

Model instances are loaded once and reused instead of initializing them for every request.

A compatible GPU can significantly improve transformer inference performance.

---

# Security & Privacy

Production deployments should consider:

- Authentication
- Authorization
- File-size limits
- MIME-type validation
- PDF validation
- Rate limiting
- HTTPS
- Secure temporary file handling
- Secret management
- Access control
- Data retention policies
- Encryption at rest and in transit

Journal entries may contain sensitive information, so application logs should avoid storing raw journal content unnecessarily.

The local-first architecture also reduces the need to send journal or document content to third-party LLM APIs.

---

# Limitations

### Model Domain Mismatch

Some pretrained models were trained on datasets different from private journal text. Domain-specific validation would improve reliability.

### Crisis Detection

The crisis component is a screening mechanism, not a clinical system. Production use would require appropriate validation and human oversight.

### Emotion Mapping

The underlying emotion model contains more categories than the application's simplified output vocabulary, so some nuanced emotions are mapped to broader categories.

### Mood Score

The mood score is an application-level interpretation and is not a clinically validated psychological measurement.

### RAG Retrieval

Generated answer quality depends heavily on retrieval quality. If relevant information is not retrieved, generation cannot reliably recover it from the document.

### PDF Extraction

The current pipeline primarily targets text-based PDFs. Scanned documents, complex tables, handwritten content, and image-heavy PDFs may require OCR or specialized document parsing.

### Local Resources

Local transformer inference can require significant RAM, CPU, or GPU resources and may be slower on CPU-only systems.

---

# Future Improvements

## Journal Analysis

- Domain-specific fine-tuning
- Mood-score calibration
- Improved emotion taxonomy
- Confidence calibration
- Batch journal analysis
- Historical mood tracking
- Temporal trend analysis

## Crisis Detection

- Larger labelled dataset
- Better contextual modelling
- Threshold calibration
- Human-in-the-loop review
- Explainability
- Safety validation

## RAG

- Hybrid BM25 + vector search
- Cross-encoder reranking
- Query rewriting
- Improved chunking strategies
- Multiple document collections
- Metadata filtering
- Persistent vector database
- Streaming responses
- Conversation-aware retrieval

## Infrastructure

- Docker deployment
- GPU inference
- Background workers
- Redis-based task queues
- Observability and metrics
- Structured logging
- Authentication
- Rate limiting
- CI/CD

---

# Production Considerations

A production architecture could separate API handling, model inference, document processing, and retrieval:

```text
                    Client
                      |
                      v
                API Gateway
                      |
                      v
                 FastAPI App
                      |
          +-----------+-----------+
          |                       |
          v                       v
    Journal Service         RAG Service
          |                       |
          v                       v
   Model Workers             Retrieval Layer
                                  |
                                  v
                            Vector Database
                                  |
                                  v
                             LLM Worker
```

For larger workloads, document indexing and model inference should be moved to dedicated workers.

---

# Reproducibility

Dependencies should be maintained in:

```text
requirements.txt
```

Model names should remain explicitly configured so the same model architecture can be reproduced across environments.

Downloaded Hugging Face models can be reused from the local cache after first initialization.

---

# Development Philosophy

The project follows these principles:

### Separation of Concerns
API, business logic, model inference, retrieval, and generation are separated.

### Replaceable Components
Models can be replaced without rewriting the complete application.

### Local-First AI
AI inference is performed locally wherever practical.

### Traceability
Model confidence, source metadata, and retrieval scores are retained where applicable.

### Defensive Design
The RAG system attempts to avoid unsupported answers, while the crisis component is treated cautiously and documented with explicit limitations.

### Production-Oriented Structure
The codebase is organized so components can later be moved into independent services or workers.

---

# Current Capabilities

```text
Journal Analysis
    |
    +-- Sentiment              ✓
    +-- Emotion                ✓
    +-- Mood Score             ✓
    +-- Crisis Risk            ✓
    +-- Summary                ✓
    |
    v
Structured JSON Response


Document RAG
    |
    +-- PDF Upload             ✓
    +-- PDF Text Extraction    ✓
    +-- Chunking               ✓
    +-- Embeddings             ✓
    +-- FAISS Index            ✓
    +-- Semantic Retrieval     ✓
    +-- Local Qwen Generation  ✓
    +-- Source Attribution     ✓
    +-- Grounded Responses     ✓
    |
    v
Answer + Sources
```

---

# Conclusion

MyManah AI demonstrates a modular approach to building a locally executable AI backend that combines NLP-based journal analysis with Retrieval-Augmented Generation.

The journal pipeline combines specialized transformer models for sentiment, emotion, crisis screening, mood interpretation, and summarization.

The document pipeline follows:

```text
PDF
 ↓
Extraction
 ↓
Chunking
 ↓
Embeddings
 ↓
FAISS
 ↓
Retrieval
 ↓
Context
 ↓
Local Qwen LLM
 ↓
Grounded Answer
```

The architecture is modular so that models, retrieval strategies, vector stores, and inference infrastructure can be upgraded independently as the system evolves.
