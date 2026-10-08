from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Local sentence-transformer embedding model.
    """

    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self):
        self.model = SentenceTransformer(
            self.MODEL_NAME
        )

    def encode(
        self,
        texts: list[str],
    ):
        """
        Convert text into normalized embedding vectors.
        """

        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

    def encode_query(self, text: str):
        """
        Generate embedding for a single query.
        """

        return self.model.encode(
            [text],
            normalize_embeddings=True,
        )