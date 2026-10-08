from app.models.sentiment import SentimentModel
from app.models.mood import MoodModel
from app.services.emotion_service import EmotionService
from app.services.crisis_service import CrisisService
from app.services.summary_service import SummaryService


class JournalService:

    def __init__(self):
        self.sentiment_model = SentimentModel()
        self.emotion_service = EmotionService()
        self.crisis_service = CrisisService()
        self.mood_model = MoodModel()
        self.summary_service = SummaryService()

    def analyze_sentiment(self, text: str) -> dict:
        return self.sentiment_model.predict(text)

    def analyze_emotion(self, text: str) -> dict:
        return self.emotion_service.predict(text)

    def analyze_crisis(self, text: str) -> dict:
        return self.crisis_service.predict(text)

    def analyze_mood(
        self,
        sentiment: str,
        emotion: str,
        confidence: float
    ) -> int:
        return self.mood_model.predict(
            sentiment,
            emotion,
            confidence
        )

    def analyze_summary(self, text: str) -> str:
        return self.summary_service.summarize(text)