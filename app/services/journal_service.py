from app.models.sentiment import SentimentModel
from app.services.emotion_service import EmotionService


class JournalService:

    def __init__(self):

        self.sentiment_model = SentimentModel()
        self.emotion_service = EmotionService()

    def analyze_sentiment(self, text: str) -> dict:

        return self.sentiment_model.predict(text)

    def analyze_emotion(self, text: str) -> dict:

        return self.emotion_service.predict(text)