
from fastapi import APIRouter

from app.schemas.journal import JournalRequest, JournalResponse
from app.services.journal_service import JournalService

router = APIRouter(tags=["Journal Analysis"])
journal_service = JournalService()


@router.post(
    "/analyze-journal",
    response_model=JournalResponse
)
def analyze_journal(request: JournalRequest):
    sentiment_result = journal_service.analyze_sentiment(request.text)  
    return JournalResponse(
        sentiment=sentiment_result["label"],
        emotion="neutral",
        moodScore=5,
        summary="Emotion, mood, crisis and summarization models are pending.",
        crisisRisk="LOW",
        confidence=sentiment_result["confidence"]
    )
