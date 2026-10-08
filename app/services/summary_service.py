from app.models.summarizer import SummarizerModel


class SummaryService:

    def __init__(self):
        self.model = SummarizerModel()

    def summarize(self, text: str) -> str:
        return self.model.predict(text)