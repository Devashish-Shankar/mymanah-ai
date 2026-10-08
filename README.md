# MyManah AI — Journal Analysis & RAG System

AI-powered journal analysis and document-grounded question answering system built for the **MyManah Generative AI & Machine Learning Engineer Intern Technical Assessment**.

The project uses **open-source Hugging Face models with local inference** and a modular Python/FastAPI architecture. The Journal Analysis pipeline is being completed first, followed by the required RAG pipeline and deployment.

> **Safety disclaimer:** This is a technical assessment prototype. Its mental-health and crisis-related outputs are experimental NLP/model outputs and must not be interpreted as a clinical diagnosis, medical advice, definitive safety assessment, or emergency-triage decision.

---

## 1. Assessment Objective

The MyManah assessment asks for an AI-powered Journal Analysis System using open-source Hugging Face models without relying on OpenAI, Gemini, Claude, or other hosted LLM APIs.

The Journal Analysis API must analyze a journal entry and return:

1. **Sentiment:** Positive / Neutral / Negative
2. **Dominant Emotion:** Happy / Sad / Anxiety / Stress / Anger / Fear / Neutral
3. **Mood Score:** 1–10
4. **AI Summary:** concise 2–3 sentence summary
5. **Crisis Risk:** LOW / MEDIUM / HIGH
6. **Confidence**

The second major requirement is a RAG pipeline:

```text
PDF → Text Extraction → Chunking → Embeddings → Vector Database
    → Retrieval → Local Generation → Grounded Answer
```

The answer must be grounded in the uploaded document rather than unrelated model knowledge.

---

## 2. Key Engineering Goals

- Local-first AI inference
- Open-source Hugging Face models
- Modular architecture
- Separation of API, service, and model layers
- Explainable application-level decision logic
- Explicit model limitations
- Input validation
- Testable components
- RAG grounding
- Production-oriented design
- Clear documentation
- Easy model replacement

The goal is not only to make the API work, but to demonstrate practical AI/ML engineering decisions.

---

## 3. Current Implementation Status

| Component | Status |
|---|---|
| Python 3.12 environment | ✅ |
| FastAPI application | ✅ |
| Uvicorn server | ✅ |
| Pydantic schemas | ✅ |
| `/health` endpoint | ✅ |
| Swagger/OpenAPI docs | ✅ |
| Local Hugging Face inference | ✅ |
| Sentiment analysis | ✅ |
| Emotion classification | ✅ |
| Crisis classifier | ✅ |
| Crisis context layer | ✅ |
| Crisis evaluation tests | ✅ |
| Mood score 1–10 | ✅ |
| `/analyze-journal` integration | ✅ |
| Overall confidence refinement | 🔧 Final cleanup |
| AI summary | 🔧 Final verification |
| PDF upload | ⏳ RAG |
| PDF extraction | ⏳ RAG |
| Text chunking | ⏳ RAG |
| Embeddings | ⏳ RAG |
| Vector database | ⏳ RAG |
| Document-grounded QA | ⏳ RAG |
| Deployment | ⏳ Final phase |
| Demo | ⏳ Final phase |

---

## 4. High-Level Architecture

### Journal Analysis

```text
                         Journal Entry
                              |
                              v
                       FastAPI Endpoint
                              |
                              v
                      JournalService
                              |
          +-------------------+-------------------+
          |                   |                   |
          v                   v                   v
     Sentiment             Emotion             Crisis
      RoBERTa              RoBERTa           ModernBERT
          |                   |                   |
          v                   v                   v
     Sentiment           Emotion Mapping     Crisis Probability
      Result                 Result                |
                                                   v
                                          Context Decision Layer
                                                   |
                                                   v
                                             LOW/MEDIUM/HIGH
          |                   |
          +---------+---------+
                    |
                    v
             Mood Score Logic
                    |
                    v
             Summary Model
                    |
                    v
              API Response
```

### Planned Complete System

```text
                         Client
                           |
                           v
                     FastAPI Backend
                           |
             +-------------+-------------+
             |                           |
             v                           v
      Journal Analysis                   RAG
             |                           |
       +-----+-----+               +-----+------+
       |     |     |               |            |
       v     v     v               v            v
   Sentiment Emotion Crisis       PDF Loader   Question
       |     |     |               |            |
       |     |     v               v            |
       |     | Context Layer    Text Chunks     |
       |     |     |               |            |
       +-----+-----+               v            |
             |                 Embeddings       |
             v                     |            |
       Mood + Summary              v            |
             |                 Vector Store     |
             +---------------------+-------------+
                                   |
                                   v
                              Final Answer
```

---

## 5. Technology Stack

### Backend
- Python 3.12
- FastAPI
- Uvicorn
- Pydantic

### Machine Learning
- PyTorch
- Hugging Face Transformers
- Hugging Face pretrained/fine-tuned transformer models

### Planned RAG
- PDF text extraction
- Sentence Transformers
- FAISS or ChromaDB
- Local Hugging Face generation model

### Development
- Git
- GitHub
- VS Code
- Python virtual environment
- Swagger/OpenAPI

---

## 6. Project Structure

```text
mymanah-ai/
│
├── app/
│   ├── main.py
│   ├── api/
│   │   └── routes/
│   │       ├── journal.py
│   │       └── rag.py
│   ├── models/
│   │   ├── sentiment.py
│   │   ├── emotion.py
│   │   ├── crisis.py
│   │   ├── mood.py
│   │   └── summarizer.py
│   ├── services/
│   │   ├── journal_service.py
│   │   ├── emotion_service.py
│   │   ├── crisis_service.py
│   │   └── summary_service.py
│   ├── rag/
│   │   ├── loader.py
│   │   ├── splitter.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   └── retriever.py
│   ├── schemas/
│   │   ├── journal.py
│   │   └── rag.py
│   └── core/
│       └── config.py
│
├── tests/
│   ├── test_sentiment.py
│   ├── test_emotion.py
│   ├── test_crisis.py
│   └── test_crisis_service.py
│
├── data/
├── README.md
├── requirements.txt
├── .gitignore
├── Dockerfile
└── .env
```

Some RAG/deployment files are introduced as implementation progresses.

---

## 7. Local-First AI Design

The assessment requires downloaded Hugging Face models and local inference.

The system does not use:

- OpenAI API
- Gemini API
- Claude API
- Hosted LLM inference APIs

Architecture:

```text
User Request
     |
     v
FastAPI
     |
     v
Python Application
     |
     v
Hugging Face Model
     |
     v
Local Inference
     |
     v
JSON Response
```

The first execution may download a model. Subsequent inference can use the local Hugging Face cache.

---

# 8. Journal Analysis API

## Endpoint

```http
POST /analyze-journal
```

## Request

```json
{
  "text": "I had a difficult day at work and I am feeling exhausted."
}
```

## Response Contract

```json
{
  "sentiment": "negative",
  "emotion": "stress",
  "moodScore": 3,
  "summary": "The user describes a difficult and exhausting day at work. The entry indicates elevated emotional stress.",
  "crisisRisk": "LOW",
  "confidence": 0.84
}
```

Exact values depend on local model inference.

---

## 8.1 Request Validation

The journal request uses Pydantic validation.

Current constraints:

```text
Minimum length: 1 character
Maximum length: 10,000 characters
```

This prevents empty requests and places a practical upper bound on one journal entry.

---

# 9. Sentiment Analysis

## Model

```text
cardiffnlp/twitter-roberta-base-sentiment-latest
```

The model returns:

```text
positive
neutral
negative
```

### Pipeline

```text
Journal Text
     |
     v
RoBERTa Sentiment Classifier
     |
     v
Label + Confidence
```

### Implementation

```text
app/models/sentiment.py
```

The model layer is responsible for loading and running inference.

The service layer integrates the result:

```text
app/services/journal_service.py
```

### Model Selection Rationale

The model was selected because:

- It is an open-source Hugging Face model.
- It directly supports sentiment classification.
- It is practical for local inference.
- It integrates cleanly with Transformers.
- It produces a confidence score.
- It avoids hosted inference APIs.

### Limitation

The model is trained for Twitter/social-media-style sentiment rather than private mental-health journal text.

Therefore it is a practical baseline, not a clinically validated journal sentiment model. A production system could improve performance through journal-domain evaluation or fine-tuning.

---

# 10. Emotion Classification

## Model

```text
SamLowe/roberta-base-go_emotions
```

The model provides a broad emotion taxonomy.

### Current Application Mapping

```text
joy          → happy
sadness      → sad
anger        → anger
fear         → fear
nervousness  → anxiety
neutral      → neutral
```

The assessment also requires `stress`. Because the underlying GoEmotions taxonomy does not map one-to-one to the required seven categories, stress requires additional application-level refinement.

### Pipeline

```text
Journal Text
     |
     v
GoEmotions / RoBERTa
     |
     v
Raw Emotion
     |
     v
Application Mapping
     |
     v
Assignment Emotion
```

Model implementation:

```text
app/models/emotion.py
```

Application mapping:

```text
app/services/emotion_service.py
```

### Why a separate emotion model?

Sentiment and emotion are different tasks.

For example:

```text
Sentiment = negative
Emotion   = fear
```

A negative sentiment alone does not identify the underlying emotional state.

A dedicated emotion classifier therefore provides richer information.

---

# 11. Crisis Risk Detection

## Model

```text
Akashpaul123/modernbert-crisis-detection
```

The model is a binary classifier:

```text
LABEL_0 → non-crisis
LABEL_1 → crisis
```

It produces a crisis probability.

However, the application does not directly map the raw probability to HIGH risk.

Instead, the probability is passed through a context-aware decision layer.

---

# 12. Crisis Context Layer

Local evaluation showed that crisis-related vocabulary can occur in different contexts.

Examples:

### Explicit denial

```text
I am not suicidal and I have no intention of hurting myself.
```

### Third-person statement

```text
My friend told me that they wanted to kill themselves.
```

### Historical statement

```text
Last year I struggled with suicidal thoughts, but I am doing much better now.
```

Therefore:

```text
Raw Model Probability
        ≠
Final Application Risk
```

The application considers:

- Explicit denial
- Third-person references
- Historical references
- Suicidal ideation
- Explicit plan/high-risk intent
- Immediate self-harm language

---

# 13. Crisis Decision Architecture

```text
                  Journal Text
                       |
                       v
             ModernBERT Classifier
                       |
                       v
             Crisis Probability
                       |
                       v
          +------------------------+
          | Context Decision Layer |
          +------------------------+
             |       |       |
             v       v       v
          Denial   Ideation  High-risk
                              intent
             |       |       |
             v       v       v
            LOW    MEDIUM    HIGH
```

The transformer remains the primary NLP signal while the context layer handles obvious application-level contextual cases.

---

# 14. Crisis Decision Policy

The intended policy is:

```text
Explicit denial
       ↓
LOW

Ideation without explicit current plan
       ↓
MEDIUM

Current explicit plan / immediate self-harm intent
       ↓
HIGH
```

The context layer is deliberately small and explainable.

---

# 15. Crisis Model Selection

Several candidate Hugging Face mental-health/crisis classifiers were evaluated locally using representative application examples.

Evaluation included:

- Normal text
- Hopelessness
- Passive ideation
- Explicit suicidal ideation
- Explicit plan
- Self-harm intent
- Third-person crisis statements
- Historical crisis statements
- Explicit denial

The selected ModernBERT crisis classifier showed useful behavior on explicit current crisis/self-harm examples during the small local evaluation.

It also showed context sensitivity in cases such as explicit denial and third-person references.

The engineering decision was therefore to keep the model and add a transparent context layer rather than continuously changing models without a domain-specific evaluation dataset.

---

# 16. Crisis Limitations

This is a prototype screening component and is not clinically validated.

### Small evaluation dataset

The current evaluation set is manually constructed and not statistically representative of real-world journal data.

### Context sensitivity

Crisis vocabulary can appear in:

- Historical statements
- Third-person discussions
- Educational content
- Supportive statements
- Explicit denials

### No clinical validation

The model has not been independently clinically validated for this application.

### Probability is not clinical risk

The classifier probability is an NLP classification output, not a calibrated probability of actual clinical risk.

### Production requirements

A production system would require:

- Properly labelled journal-domain data
- Independent validation
- Threshold calibration
- False-positive/false-negative analysis
- Subgroup/fairness evaluation
- Human review
- Clinical/safety review
- Post-deployment monitoring

---

# 17. Overall Confidence

The API `confidence` field is treated as an **overall model-confidence indicator**, not clinical probability.

The intended calculation is:

```text
Overall Confidence =
    (Sentiment Confidence + Emotion Confidence) / 2
```

The raw crisis probability is intentionally not used as the overall confidence.

Reason:

```text
Crisis Model
     ↓
Crisis Probability
     ↓
Context Decision Layer
     ↓
LOW / MEDIUM / HIGH
```

Because an additional application-level decision layer changes the final crisis result, directly exposing the raw crisis probability as overall confidence can be misleading.

The confidence value should not be interpreted as:

- Clinical certainty
- Probability of self-harm
- Probability of a diagnosis
- Medical risk probability

It is an application-level confidence indicator for the underlying sentiment/emotion analysis.

---

# 18. Mood Score

## Requirement

```text
1 = Very Poor Mood
10 = Excellent Mood
```

## Design

The current prototype uses an explainable application-level scoring layer based on sentiment and emotion rather than adding an unnecessary independent model.

```text
Journal Text
     |
     +------> Sentiment
     |
     +------> Emotion
     |
     v
Mood Scoring Layer
     |
     v
1–10
```

The approach is:

```text
Positive + Happy      → High mood
Positive + Neutral    → Moderately high mood
Neutral               → Middle range
Negative + Sad        → Low mood
Negative + Anxiety    → Low mood
Negative + Stress     → Low mood
Negative + Fear       → Low mood
Negative + Anger      → Lower/middle range
```

This is an engineering baseline, not a clinically validated mood scale.

A future production version could replace this layer with a trained regression model using a properly labelled mood dataset.

---

# 19. AI Summary

The assessment requires a concise 2–3 sentence AI summary.

The project uses a local Hugging Face summarization model rather than a hosted LLM API.

Planned/implemented layers:

```text
app/models/summarizer.py
app/services/summary_service.py
```

Flow:

```text
Journal Text
     |
     v
Local Summarization Model
     |
     v
Concise Summary
```

For very short journal entries, forcing abstractive summarization can produce unnecessary or low-quality output, so short inputs can be handled separately.

The summary component is being verified before the final journal-analysis phase is frozen.

---

# 20. Complete Journal Analysis Flow

```text
POST /analyze-journal
          |
          v
     Validate Input
          |
          v
    JournalService
          |
   +------+------+------+------+
   |      |      |      |
   v      v      v      v
Sentiment Emotion Crisis  Summary
   |      |      |      |
   |      |      v      |
   |      | Context     |
   |      | Layer       |
   |      |      |      |
   +------+------+------+
          |
          v
      Mood Score
          |
          v
  Overall Confidence
          |
          v
      JSON Response
```

---

# 21. API Usage

Start the server:

```bash
uvicorn app.main:app --reload
```

Application:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

### Example Request

```json
{
  "text": "Today was difficult at work. I had several deadlines and felt overwhelmed throughout the afternoon. I was worried that I would not finish everything on time. After coming home, I talked with my family and felt a little calmer. I am still tired, but I think tomorrow will be better."
}
```

### Example Response

```json
{
  "sentiment": "negative",
  "emotion": "stress",
  "moodScore": 3,
  "summary": "The user describes a stressful and overwhelming workday with several deadlines. Support from family helped the user feel somewhat calmer despite ongoing tiredness.",
  "crisisRisk": "LOW",
  "confidence": 0.84
}
```

Exact values depend on model inference.

---

# 22. Swagger / OpenAPI

FastAPI automatically exposes interactive documentation:

```text
http://127.0.0.1:8000/docs
```

An evaluator can:

1. Open Swagger.
2. Select `POST /analyze-journal`.
3. Click **Try it out**.
4. Enter journal text.
5. Execute the request.
6. Inspect the JSON response.

This provides a simple demonstration interface without requiring a separate frontend.

---

# 23. Testing Strategy

## Sentiment

Test:

- Positive
- Negative
- Neutral

## Emotion

Test:

- Happy
- Sad
- Anxiety
- Stress
- Anger
- Fear
- Neutral

## Crisis

Test:

- Normal text
- General hopelessness
- Passive ideation
- Explicit ideation
- Explicit plan
- Immediate self-harm intent
- Third-person reference
- Historical reference
- Explicit denial

Example:

```bash
python -m tests.test_crisis_service
```

Running the test as a module from the project root ensures the `app` package is correctly resolved.

---

# 24. API and Code Quality

The project follows:

```text
API Route
    ↓
Service Layer
    ↓
Model Layer
```

### API layer

Responsible for:

- HTTP endpoints
- Request validation
- Response serialization

### Service layer

Responsible for:

- Business logic
- Combining model results
- Crisis context logic
- Mood scoring
- Summary orchestration

### Model layer

Responsible for:

- Model loading
- Local inference
- Raw model outputs

This separation improves:

- Maintainability
- Testability
- Model replacement
- Debugging
- Code readability

---

# 25. Design Decisions

## Why FastAPI?

FastAPI provides:

- Automatic OpenAPI documentation
- Strong request validation
- Python ML integration
- Simple REST API development
- Easy deployment

## Why Hugging Face Transformers?

Because the assessment explicitly requires open-source Hugging Face models and local inference.

## Why specialized models?

Sentiment, emotion, crisis detection, and summarization are different NLP tasks. Specialized models allow each task to use an appropriate model.

## Why a service layer?

To keep business logic independent of HTTP routing and model implementation.

## Why a crisis context layer?

Because raw crisis probability can be affected by negation, historical context, or third-person references.

## Why application-level mood scoring?

It is transparent, deterministic, and avoids adding another model before a suitable labelled mood dataset exists.

---

# 26. Installation

## Clone

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd mymanah-ai
```

## Create Virtual Environment

Windows:

```powershell
python -m venv .venv
```

Activate:

```powershell
.venv\Scripts\Activate.ps1
```

## Install Dependencies

```powershell
pip install -r requirements.txt
```

Current core dependencies:

```text
fastapi
uvicorn
pydantic
python-dotenv
torch
transformers
```

RAG-specific dependencies will be added when the RAG implementation is completed.

---

# 27. Run Locally

```powershell
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

Health:

```http
GET /health
```

---

# 28. Hugging Face Model Cache

The first model execution may download model files.

After download, Transformers uses the local cache.

This supports the local-inference requirement.

For deployment, models can be pre-downloaded during image build or startup to make deployment behavior more predictable.

---

# 29. Security and Privacy

Journal entries may contain highly sensitive information.

A production implementation should consider:

- HTTPS
- Authentication
- Authorization
- Encryption in transit
- Encryption at rest
- Secure file uploads
- Access control
- Data retention policies
- Data deletion
- Minimal logging
- Avoiding raw journal text in logs

The current assessment prototype is not presented as a complete clinical-data security system.

---

# 30. Scalability and Production Thinking

## Model Lifecycle

Load models once during application startup instead of reloading them per request.

## GPU

Use CUDA when available for supported models.

## Quantization

Quantized models can reduce:

- RAM
- Latency
- Infrastructure cost

## Batch Processing

Multiple journal entries can potentially be processed in batches.

## Background Workers

Long-running summarization/RAG operations can be moved to background workers.

## Caching

Potential caching targets:

- Model cache
- Embedding cache
- Document processing
- Repeated RAG queries

## Vector Database

FAISS is suitable for a lightweight local implementation. A production-scale deployment could use a dedicated vector database.

## Horizontal Scaling

A future architecture could separate:

```text
Load Balancer
      |
+-----+-----+
|     |     |
API   API   API
      |
Inference Layer
      |
Vector Database
```

---

# 31. RAG — Required Second Component

The second major assessment requirement is Retrieval-Augmented Generation.

Target pipeline:

```text
                 PDF Upload
                     |
                     v
               PDF Extraction
                     |
                     v
                Text Cleaning
                     |
                     v
                  Chunking
                     |
                     v
                Embeddings
                     |
                     v
               Vector Store
                     |
                     v
               User Question
                     |
                     v
             Query Embedding
                     |
                     v
             Similarity Search
                     |
                     v
             Relevant Chunks
                     |
                     v
          Local Hugging Face Model
                     |
                     v
                  Answer
```

---

# 32. RAG Requirements

The implementation will support:

1. PDF upload
2. PDF text extraction
3. Text chunking
4. Embedding generation
5. Vector storage
6. Similarity retrieval
7. Document-grounded answer generation

If the document does not contain enough information, the system should avoid inventing an unsupported answer.

---

# 33. Planned RAG Components

### PDF Loader

```text
app/rag/loader.py
```

Responsibilities:

- Read uploaded PDF
- Extract text
- Validate document

### Text Splitter

```text
app/rag/splitter.py
```

Responsibilities:

- Split text into retrieval chunks
- Maintain contextual overlap

### Embedding Model

```text
app/rag/embeddings.py
```

Responsibilities:

- Convert chunks to vectors
- Convert questions to vectors

### Vector Store

```text
app/rag/vector_store.py
```

Responsibilities:

- Store embeddings
- Similarity search
- Maintain document metadata

### Retriever

```text
app/rag/retriever.py
```

Responsibilities:

- Embed query
- Search vector store
- Return relevant chunks

---

# 34. RAG Grounding Strategy

The generation prompt will be structured around retrieved context:

```text
You are answering a question using the uploaded document.

Context:
<retrieved document chunks>

Question:
<user question>

Instructions:
- Answer using only the supplied context.
- Do not invent information.
- If the answer is not present in the context, say that the document does not provide enough information.
```

This explicitly enforces document grounding.

---

# 35. RAG Evaluation

The RAG system will be evaluated for:

### Retrieval quality

Does it retrieve the relevant document sections?

### Groundedness

Does the answer correspond to retrieved text?

### Missing information

Does it avoid hallucinating when the answer is absent?

### Chunking

Does the chunk size preserve sufficient context?

### Latency

How long does:

```text
Question → Retrieval → Generation
```

take locally?

---

# 36. Deployment Plan

The assessment allows either a hosted deployment link, Loom video, or screen recording.

The preferred final deliverable is a hosted API with Swagger.

Target:

```text
GitHub
   ↓
Deployment Platform
   ↓
FastAPI
   ↓
Local Hugging Face Models
   ↓
Public API
   ↓
/docs
```

Deployment must account for:

- RAM
- CPU/GPU
- Model download size
- Startup time
- Cold starts
- Persistent model cache
- Request timeouts

Deployment will be completed after RAG is functional.

---

# 37. Demo Plan

## Journal Analysis

```text
Open Swagger
   ↓
POST /analyze-journal
   ↓
Enter journal
   ↓
Execute
   ↓
Show:
- sentiment
- emotion
- moodScore
- summary
- crisisRisk
- confidence
```

## RAG

```text
Upload PDF
   ↓
Process document
   ↓
Ask question
   ↓
Retrieve relevant chunks
   ↓
Generate grounded answer
```

The demo should briefly explain model choices, local inference, context-aware crisis handling, RAG grounding, and limitations.

---

# 38. Assignment Requirement Mapping

| Assessment Requirement | Status |
|---|---|
| Hugging Face models | ✅ |
| Local model download | ✅ |
| Local inference | ✅ |
| No OpenAI | ✅ |
| No Gemini | ✅ |
| No Claude | ✅ |
| FastAPI | ✅ |
| Sentiment | ✅ |
| Emotion | ✅ |
| Mood Score 1–10 | ✅ |
| AI Summary 2–3 sentences | 🔧 Final verification |
| Crisis LOW/MEDIUM/HIGH | ✅ |
| Confidence | 🔧 Final refinement |
| PDF upload | ⏳ |
| PDF extraction | ⏳ |
| Text splitting | ⏳ |
| Embeddings | ⏳ |
| Vector database | ⏳ |
| Document-grounded QA | ⏳ |
| Architecture overview | ✅ |
| Model selection rationale | ✅ |
| Installation steps | ✅ |
| API usage examples | ✅ |
| Design decisions | ✅ |
| Limitations | ✅ |
| Demo | ⏳ |
| Deployment | ⏳ |

---

# 39. Evaluation Criteria Alignment

## AI/ML Understanding — 25%

Demonstrated through:

- Task-specific transformer selection
- Local inference
- Sentiment vs emotion distinction
- Crisis model + context layer
- Embedding/RAG architecture
- Model limitations
- Confidence interpretation

## Model Selection & Justification — 15%

Each major model is documented with:

- Task
- Model name
- Selection rationale
- Local inference considerations
- Limitations

## Code Quality — 15%

Demonstrated through:

- Layered architecture
- Model/service separation
- Pydantic schemas
- Reusable services
- Testable components
- Explicit boundaries

## API Design — 10%

Demonstrated through:

- FastAPI
- REST endpoint
- Typed schemas
- Validation
- Swagger/OpenAPI

## RAG Implementation — 15%

Target:

```text
PDF
→ extraction
→ chunking
→ embeddings
→ vector store
→ retrieval
→ local generation
```

## Documentation — 10%

This README covers:

- Architecture
- Models
- Installation
- API usage
- Design decisions
- Testing
- Limitations
- Production considerations

## Scalability & Production Thinking — 10%

Covered through:

- Model lifecycle
- GPU
- Quantization
- Batch processing
- Caching
- Background workers
- Vector DB scaling
- Security/privacy
- Deployment

---

# 40. Assumptions

1. Journal text is provided as UTF-8 text.
2. A single journal request is limited to 10,000 characters.
3. Hugging Face model downloads are available during initial setup.
4. CPU inference is acceptable for local development.
5. Crisis predictions are screening/application signals only.
6. Mood scoring is an engineering baseline, not a validated clinical scale.
7. RAG answers must be grounded in uploaded documents.
8. Deployment infrastructure will provide enough resources for the selected models.

---

# 41. Known Limitations

## ML

- Sentiment model domain differs from journal-domain data.
- Emotion mapping does not perfectly correspond to the assessment categories.
- Crisis classifier is context-sensitive.
- Crisis model is not clinically validated for this application.
- Current evaluation data is small and manually constructed.
- Mood scoring is application-based rather than trained on labelled mood data.
- Summary quality depends on the selected local model.

## Engineering

- Initial model download can be slow.
- CPU inference can be slower than GPU inference.
- Transformer models can require significant RAM.
- Model startup can increase deployment cold-start time.
- Authentication is not currently implemented.
- Production observability is limited.
- RAG quality depends on chunking and embedding quality.

## Safety

The system should not be used as:

- A medical diagnostic system
- A clinical decision-support system
- A definitive suicide-risk assessment
- An emergency-response system
- A replacement for professional mental-health care

Production use would require clinical, safety, privacy, legal, and regulatory review.

---

# 42. Future Improvements

### Journal Analysis

- Journal-domain fine-tuning
- Better stress classification
- Dedicated mood regression
- Confidence calibration
- Improved crisis context classification
- Multilingual support

### RAG

- Hybrid BM25 + vector retrieval
- Reranking
- Better chunking
- Page/source citations
- Multi-document retrieval
- Persistent vector database

### Deployment

- Docker
- GPU inference
- Quantization
- Model serving
- Authentication
- Rate limiting
- Monitoring
- CI/CD

---

# 43. Development Roadmap

```text
Project Setup
      ↓
Sentiment
      ↓
Emotion
      ↓
Crisis + Context Layer
      ↓
Mood Score
      ↓
AI Summary
      ↓
Complete Journal API
      ↓
RAG
  ├── PDF
  ├── Extraction
  ├── Chunking
  ├── Embeddings
  ├── Vector Store
  ├── Retrieval
  └── Generation
      ↓
Testing
      ↓
Deployment
      ↓
Demo
      ↓
Submission
```

---

# 44. Submission Checklist

## Source Code

- [ ] Complete implementation
- [ ] GitHub repository
- [ ] `requirements.txt`
- [ ] Clean project structure

## Journal Analysis

- [x] Sentiment
- [x] Emotion
- [x] Mood Score
- [ ] Final AI Summary verification
- [x] Crisis risk
- [ ] Final overall confidence verification

## RAG

- [ ] PDF upload
- [ ] PDF extraction
- [ ] Chunking
- [ ] Embeddings
- [ ] Vector database
- [ ] Retrieval
- [ ] Document-grounded generation

## Documentation

- [x] Architecture overview
- [x] Model selection rationale
- [x] Installation
- [x] API examples
- [x] Design decisions
- [x] Limitations
- [x] Assumptions
- [x] Production considerations

## Demo

- [ ] Hosted deployment URL OR
- [ ] Loom video OR
- [ ] Screen recording

## Final Submission

- [ ] GitHub repository link
- [ ] Demo/deployment link
- [ ] Brief approach explanation
- [ ] Assumptions
- [ ] Limitations

---

# 45. Brief Approach for Submission

The project uses a modular local-AI architecture built with FastAPI and Hugging Face Transformers.

For journal analysis, separate transformer models are used for sentiment, emotion, and crisis detection. Crisis detection additionally uses a lightweight context-aware decision layer to handle explicit denial, historical statements, and third-person references. Mood score is generated through an explainable application-level scoring layer, while summarization uses a local Hugging Face model.

For RAG, uploaded PDFs are processed into text chunks, converted into embeddings, stored in a vector database, and retrieved for user questions. A local generation model then produces an answer grounded in the retrieved document context.

The architecture separates API routes, services, model inference, and RAG components so individual models and infrastructure components can be replaced without rewriting the entire system.

---

# 46. Final Architecture Summary

```text
                         MYMANAH AI
                             |
             +---------------+---------------+
             |                               |
             v                               v
       JOURNAL ANALYSIS                     RAG
             |                               |
      +------+------+                    +---+---+
      |      |      |                    |       |
      v      v      v                    v       v
 Sentiment Emotion Crisis              PDF    Question
      |      |      |                    |       |
      |      |   Context                 v       |
      |      |    Layer              Chunks      |
      |      |      |                    |       |
      +------+------+                    v       |
             |                       Embeddings |
             v                           |       |
         Mood Score                      v       |
             |                       Vector DB   |
             v                           |       |
          Summary                        v       |
             +---------------------------+-------+
                                         |
                                         v
                                  Local Generation
                                         |
                                         v
                                    Final Answer
```

---

# 47. Conclusion

MyManah AI is being developed as a local-first, modular AI engineering system combining transformer-based journal analysis with Retrieval-Augmented Generation.

The current implementation demonstrates:

- NLP classification
- Transformer inference
- Hugging Face model selection
- Local AI inference
- FastAPI API design
- Context-aware application logic
- Explainable scoring
- Model limitation analysis
- Testing strategy
- RAG system design
- Production considerations

The immediate implementation priority is:

```text
Final Journal Verification
          ↓
RAG Implementation
          ↓
Final Testing
          ↓
Deployment
          ↓
Demo
          ↓
Submission
```

---

## Safety Disclaimer

This project is an engineering prototype created for the MyManah technical assessment. It is not a medical device, clinical decision-support system, diagnostic system, emergency response system, or substitute for professional mental-health care. Crisis-related outputs are experimental model/application outputs and must not be treated as definitive assessments of a person's safety.
