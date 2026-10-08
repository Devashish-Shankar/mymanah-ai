
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
    emotion_result = journal_service.analyze_emotion(request.text)  # Currently not used in response
    crisis_risk_result = journal_service.analyze_crisis(request.text)  # Currently not used in response
    return JournalResponse(
        sentiment=sentiment_result["label"],
        emotion=emotion_result["emotion"],
        moodScore=5,
        summary="Mood and summarization models are pending.",
        crisisRisk=crisis_risk_result["risk_level"],
        confidence=crisis_risk_result["confidence"]
    )
