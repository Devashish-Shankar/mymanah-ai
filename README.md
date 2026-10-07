# MyManah AI

Local AI-powered Journal Analysis and Retrieval-Augmented Generation (RAG) system built for the MyManah Generative AI & Machine Learning Engineer Intern technical assessment.

## Project Goals

The system will provide:

- Journal sentiment analysis
- Emotion classification
- Mood scoring
- AI-generated journal summaries
- Crisis risk classification
- PDF document ingestion
- Document embeddings
- Vector search
- Retrieval-Augmented Generation (RAG)
- Local Hugging Face model inference

## Technology Stack

- Python
- FastAPI
- PyTorch
- Hugging Face Transformers
- Sentence Transformers
- FAISS
- PyMuPDF

## Current Status

### Completed

- Project structure
- Python virtual environment
- FastAPI application
- Health-check endpoint
- Journal request/response schemas
- `/analyze-journal` API endpoint
- API validation
- Swagger documentation

### In Progress

- Local Hugging Face sentiment model
- Emotion classification
- Crisis risk detection
- Mood scoring
- Local summarization
- PDF RAG pipeline

## Running Locally

Create and activate the Python virtual environment:

```bash
python -m venv .venv
```

Activate it and install dependencies:

```bash
pip install -r requirements.txt
```

Run the API:

```bash
python -m uvicorn app.main:app --reload
```

Open the API documentation:

```text
http://127.0.0.1:8000/docs
```

## Architecture

The project is being developed as a modular AI backend consisting of:

```text
FastAPI
   |
   +-- Journal Analysis
   |      |
   |      +-- Sentiment
   |      +-- Emotion
   |      +-- Crisis Risk
   |      +-- Mood Score
   |      +-- Summarization
   |
   +-- RAG
          |
          +-- PDF Loader
          +-- Text Chunking
          +-- Embeddings
          +-- FAISS
          +-- Retrieval
          +-- Local LLM
```

## Important

The current journal endpoint uses a temporary mock response. Actual Hugging Face model inference will be integrated in the next development phase.