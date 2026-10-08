from transformers import pipeline


class SummarizerModel:
    """
    Local Hugging Face summarization model.
    """

    MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

    def __init__(self):
        self.pipeline = pipeline(
            "summarization",
            model=self.MODEL_NAME
        )

    def predict(self, text: str) -> str:
        # Handle short journal entries safely.
        if len(text.split()) < 20:
            return text.strip()

        word_count = len(text.split())

        max_length = min(100, max(30, word_count // 2))
        min_length = min(30, max(10, word_count // 4))

        if min_length >= max_length:
            min_length = max(5, max_length - 5)

        result = self.pipeline(
            text,
            min_length=min_length,
            max_length=max_length,
            do_sample=False
        )

        return result[0]["summary_text"].strip()