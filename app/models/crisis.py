from transformers import pipeline


class CrisisModel:
    """
    Local crisis-risk classifier using a fine-tuned ModernBERT model.

    LABEL_0 = non-crisis
    LABEL_1 = crisis

    Note:
    This model is used as a screening signal only.
    It is not a clinical diagnostic or triage system.
    """

    MODEL_NAME = "Akashpaul123/modernbert-crisis-detection"

    def __init__(self):
        self.pipeline = pipeline(
            "text-classification",
            model=self.MODEL_NAME,
            top_k=None
        )

    def predict(self, text: str) -> dict:
        results = self.pipeline(text)[0]

        probabilities = {
            result["label"]: float(result["score"])
            for result in results
        }

        crisis_probability = probabilities.get(
            "LABEL_1",
            0.0
        )

        return {
            "crisis_probability": crisis_probability,
            "non_crisis_probability": probabilities.get(
                "LABEL_0",
                0.0
            )
        }