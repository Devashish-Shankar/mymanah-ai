
from fastapi import APIRouter

from app.schemas.journal import JournalRequest, JournalResponse

router = APIRouter(tags=["Journal Analysis"])


@router.post(
    "/analyze-journal",
    response_model=JournalResponse
)
def analyze_journal(request: JournalRequest):
    # Temporary mock response for API testing.
    # No AI inference happens here yet.
    return JournalResponse(
        sentiment="neutral",
        emotion="neutral",
        moodScore=5,
        summary="Model integration is pending.",
        crisisRisk="LOW",
        confidence=0.0
    )
