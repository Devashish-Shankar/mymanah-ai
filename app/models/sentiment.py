from transformers import pipeline


class SentimentModel:
    def __init__(self):
        self.pipeline = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )

    def predict(self, text: str) -> dict:
        result = self.pipeline(text)[0]

        return {
            "label": result["label"].lower(),
            "confidence": float(result["score"])
        }