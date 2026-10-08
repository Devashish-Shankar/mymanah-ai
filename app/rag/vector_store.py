import faiss
import numpy as np


class FAISSVectorStore:
    """
    FAISS vector store using normalized inner product.

    Since embeddings are L2-normalized, inner product is equivalent
    to cosine similarity.
    """

    def __init__(self):
        self.index = None
        self.chunks = []

    def build(
        self,
        embeddings,
        chunks: list[dict],
    ):
        """
        Build a FAISS index from embeddings.
        """

        embeddings = np.asarray(
            embeddings,
            dtype="float32",
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array."
            )

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Number of embeddings must match number of chunks."
            )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(embeddings)

        self.chunks = chunks

    def search(
        self,
        query_embedding,
        top_k: int = 5,
    ) -> list[dict]:

        if self.index is None:
            raise RuntimeError(
                "FAISS index has not been built."
            )

        query_embedding = np.asarray(
            query_embedding,
            dtype="float32",
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(
                1,
                -1,
            )

        scores, indices = self.index.search(
            query_embedding,
            min(top_k, len(self.chunks)),
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0],
        ):
            if index < 0:
                continue

            chunk = self.chunks[index].copy()

            chunk["score"] = float(score)

            results.append(chunk)

        return results