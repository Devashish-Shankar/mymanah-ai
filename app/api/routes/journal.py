
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
    crisis_risk_result = journal_service.analyze_crisis(request.text) 
    mood_score = journal_service.analyze_mood(
        sentiment=sentiment_result["label"],
        emotion=emotion_result["emotion"],
        confidence=sentiment_result["confidence"]
    )
    summary = journal_service.analyze_summary(request.text)
    overall_confidence = round(
        (
            sentiment_result["confidence"]
            + emotion_result["confidence"]
        ) / 2,
        4
    )
    return JournalResponse(
        sentiment=sentiment_result["label"],
        emotion=emotion_result["emotion"],
        moodScore=mood_score,
        summary=summary,
        crisisRisk=crisis_risk_result["crisisRisk"],
        confidence=overall_confidence
    )

