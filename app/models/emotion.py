from transformers import pipeline


class EmotionModel:

    def __init__(self):

        self.pipeline = pipeline(
            "text-classification",
            model="SamLowe/roberta-base-go_emotions",
            top_k=None
        )

    def predict(self, text: str) -> dict:

        results = self.pipeline(text)[0]

        results = sorted(
            results,
            key=lambda x: x["score"],
            reverse=True
        )

        dominant = results[0]

        return {
            "label": dominant["label"].lower(),
            "confidence": float(dominant["score"]),
            "all_emotions": results
        }