
from fastapi import FastAPI

from app.api.routes.journal import router as journal_router

app = FastAPI(
    title="MyManah AI Journal Analysis API",
    description="Local AI-powered journal analysis and RAG system",
    version="1.0.0"
)

app.include_router(journal_router)

@app.get("/")
def root():
    return {
        "message": "Welcome to MyManah AI",
        "status": "running"
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }
