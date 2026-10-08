class MoodModel:
    """
    Explainable mood scoring layer.

    Combines sentiment, emotion and confidence signals
    to produce a score from 1 to 10.
    """

    def predict(
        self,
        sentiment: str,
        emotion: str,
        confidence: float
    ) -> int:

        sentiment = sentiment.lower()
        emotion = emotion.lower()

        # Strong positive states
        if sentiment == "positive":
            if emotion in ["happy", "joy"]:
                return 9
            if emotion == "neutral":
                return 8
            return 8

        # Neutral states
        if sentiment == "neutral":
            if emotion in ["sad", "anxiety", "stress", "fear", "anger"]:
                return 5
            return 6

        # Negative states
        if sentiment == "negative":
            if emotion in ["sad", "fear"]:
                return 3
            if emotion in ["anxiety", "stress"]:
                return 3
            if emotion == "anger":
                return 4
            return 4

        return 5