from app.models.sentiment import SentimentModel


class JournalService:
    def __init__(self):
        self.sentiment_model = SentimentModel()

    def analyze_sentiment(self, text: str) -> dict:
        return self.sentiment_model.predict(text)