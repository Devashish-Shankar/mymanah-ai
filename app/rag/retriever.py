from app.rag.embeddings import EmbeddingModel
from app.rag.vector_store import FAISSVectorStore


class RAGRetriever:
    """
    Semantic document retriever.

    Query
      ↓
    Query embedding
      ↓
    FAISS similarity search
      ↓
    Similarity threshold
      ↓
    Relevant document chunks
    """

    MIN_SIMILARITY = 0.40

    def __init__(self):
        self.embedding_model = EmbeddingModel()
        self.vector_store = FAISSVectorStore()

    def build_index(
        self,
        chunks: list[dict],
    ):
        """
        Generate embeddings and build the FAISS index.
        """

        if not chunks:
            raise ValueError(
                "Cannot build RAG index without chunks."
            )

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = self.embedding_model.encode(
            texts
        )

        self.vector_store.build(
            embeddings=embeddings,
            chunks=chunks,
        )

    def retrieve(
        self,
        question: str,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Retrieve semantically relevant chunks.

        Chunks below MIN_SIMILARITY are rejected so that
        the generation model does not receive unrelated
        document content.
        """

        if not question.strip():
            return []

        query_embedding = (
            self.embedding_model.encode_query(
                question
            )
        )

        results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=top_k,
        )

        filtered_results = [
            result
            for result in results
            if result["score"] >= self.MIN_SIMILARITY
        ]

        return filtered_results