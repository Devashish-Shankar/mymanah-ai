from app.models.emotion import EmotionModel


class EmotionService:

    EMOTION_MAPPING = {  # noqa: RUF012
        "joy": "happy",
        "sadness": "sad",
        "anger": "anger",
        "fear": "fear",
        "nervousness": "anxiety",
        "neutral": "neutral",
    }

    def __init__(self):

        self.model = EmotionModel()

    def predict(self, text: str) -> dict:

        result = self.model.predict(text)

        raw_label = result["label"]

        mapped_label = self.EMOTION_MAPPING.get(
            raw_label,
            "neutral"
        )

        return {
            "emotion": mapped_label,
            "confidence": result["confidence"],
            "raw_emotion": raw_label,
        }