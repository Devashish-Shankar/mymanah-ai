from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class SummarizerModel:
    """
    Local Hugging Face summarization model.
    """

    MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.MODEL_NAME
        )

    def predict(self, text: str) -> str:

        # For very short journal entries, avoid forcing
        # an abstractive summarization model.
        if len(text.split()) < 20:
            return text.strip()

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        summary_ids = self.model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=100,
            min_length=25,
            num_beams=4,
            early_stopping=True
        )

        summary = self.tokenizer.decode(
            summary_ids[0],
            skip_special_tokens=True
        )

        return summary.strip()