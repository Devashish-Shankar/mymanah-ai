from app.models.sentiment import SentimentModel
from app.services.emotion_service import EmotionService
from app.services.crisis_service import CrisisService


class JournalService:

    def __init__(self):
        self.sentiment_model = SentimentModel()
        self.emotion_service = EmotionService()
        self.crisis_service = CrisisService()

    def analyze_sentiment(self, text: str) -> dict:
        return self.sentiment_model.predict(text)

    def analyze_emotion(self, text: str) -> dict:
        return self.emotion_service.predict(text)

    def analyze_crisis(self, text: str) -> dict:
        return self.crisis_service.predict(text)