from app.rag.loader import PDFLoader
from app.rag.splitter import TextSplitter
from app.rag.retriever import RAGRetriever
from app.rag.generator import RAGGenerator


PDF_PATH = "data/sample.pdf"


def main():

    # --------------------------------------------------
    # 1. LOAD PDF
    # --------------------------------------------------

    print("=" * 70)
    print("LOADING PDF")
    print("=" * 70)

    loader = PDFLoader()

    pages = loader.load(
        PDF_PATH
    )

    print(
        f"Pages extracted: {len(pages)}"
    )

    # --------------------------------------------------
    # 2. SPLIT DOCUMENT
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("CREATING CHUNKS")
    print("=" * 70)

    splitter = TextSplitter(
        chunk_size=800,
        chunk_overlap=150,
    )

    chunks = splitter.split(
        pages
    )

    print(
        f"Chunks created: {len(chunks)}"
    )

    # --------------------------------------------------
    # 3. BUILD RETRIEVER
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("BUILDING RAG INDEX")
    print("=" * 70)

    retriever = RAGRetriever()

    retriever.build_index(
        chunks
    )

    print(
        "FAISS index built successfully."
    )

    # --------------------------------------------------
    # 4. LOAD GENERATION MODEL
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("LOADING GENERATION MODEL")
    print("=" * 70)

    generator = RAGGenerator()

    # --------------------------------------------------
    # 5. TEST QUESTIONS
    # --------------------------------------------------

    questions = [
        "What is deep learning?",
        "What are neural networks?",
        "What is machine learning?",
        "What is the capital of France?",
    ]

    for question in questions:

        print("\n" + "=" * 70)
        print(
            f"QUESTION: {question}"
        )
        print("=" * 70)

        # Retrieve relevant chunks.
        results = retriever.retrieve(
            question,
            top_k=5,
        )

        # Generate answer from retrieved context.
        answer = generator.generate(
            question=question,
            retrieved_chunks=results,
        )

        print("\nANSWER:")
        print(answer)

        print("\nSOURCES:")

        for result in results[:3]:

            print(
                f"- Page {result['page']} | "
                f"Chunk {result['chunk_id']} | "
                f"Similarity {result['score']:.4f}"
            )


if __name__ == "__main__":
    main()