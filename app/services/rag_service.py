from app.rag.loader import PDFLoader
from app.rag.splitter import TextSplitter
from app.rag.retriever import RAGRetriever
from app.rag.generator import RAGGenerator


class RAGService:
    """
    End-to-end Retrieval-Augmented Generation service.

    Pipeline:

    PDF
      ↓
    Text Extraction
      ↓
    Chunking
      ↓
    Embeddings
      ↓
    FAISS Retrieval
      ↓
    Local Qwen Generation
      ↓
    Answer + Sources
    """

    def __init__(self):
        self.loader = PDFLoader()
        self.splitter = TextSplitter()
        self.retriever = RAGRetriever()
        self.generator = RAGGenerator()

        self.is_ready = False
        self.pages_count = 0
        self.chunks_count = 0

    def ingest_pdf(self, pdf_path: str) -> dict:
        """
        Load a PDF, split it into chunks and build
        the FAISS vector index.
        """

        pages = self.loader.load(pdf_path)

        if not pages:
            raise ValueError(
                "No text could be extracted from the PDF."
            )

        chunks = self.splitter.split(pages)

        if not chunks:
            raise ValueError(
                "No chunks were generated from the PDF."
            )

        self.retriever.build_index(chunks)

        self.pages_count = len(pages)
        self.chunks_count = len(chunks)
        self.is_ready = True

        return {
            "pages": self.pages_count,
            "chunks": self.chunks_count,
        }

    def ask(self, question: str) -> dict:
        """
        Retrieve relevant chunks and generate an answer
        strictly from the retrieved document context.
        """

        if not self.is_ready:
            raise RuntimeError(
                "No document has been uploaded yet."
            )

        # Retrieve relevant document chunks
        retrieved_chunks = self.retriever.retrieve(
            question,
            top_k=3
        )

        if not retrieved_chunks:
            return {
                "answer": (
                    "The answer is not available "
                    "in the provided document."
                ),
                "sources": [],
            }

        # Generate answer using the SAME retrieved chunks.
        # RAGGenerator itself constructs the context.
        answer = self.generator.generate(
            question=question,
            retrieved_chunks=retrieved_chunks,
        )

        # Build source information
        sources = []

        for chunk in retrieved_chunks:

            sources.append(
                {
                    "page": chunk.get("page", 0),
                    "chunk": chunk.get(
                        "chunk",
                        chunk.get(
                            "chunk_id",
                            chunk.get(
                                "chunk_index",
                                chunk.get("index", 0)
                            )
                        )
                    ),
                    "similarity": round(
                        float(
                            chunk.get(
                                "similarity",
                                chunk.get(
                                    "score",
                                    0.0
                                )
                            )
                        ),
                        4
                    ),
                }
            )

        return {
            "answer": answer,
            "sources": sources,
        }


# Shared service instance
rag_service = RAGService()