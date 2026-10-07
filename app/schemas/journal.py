
from typing import Literal

from pydantic import BaseModel, Field


class JournalRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="User's journal entry"
    )


class JournalResponse(BaseModel):
    sentiment: Literal[
        "positive", "neutral", "negative"
    ]

    emotion: Literal[
        "happy", "sad", "anxiety",
        "stress", "anger", "fear", "neutral"
    ]

    moodScore: int = Field(ge=1, le=10)

    summary: str

    crisisRisk: Literal["LOW", "MEDIUM", "HIGH"]

    confidence: float = Field(ge=0.0, le=1.0)
