from app.rag.loader import PDFLoader
from app.rag.splitter import TextSplitter
from app.rag.retriever import RAGRetriever


PDF_PATH = "data/sample.pdf"


def main():
    # --------------------------------------------------
    # STEP 1: LOAD PDF
    # --------------------------------------------------

    print("=" * 70)
    print("STEP 1: PDF LOADING")
    print("=" * 70)

    loader = PDFLoader()

    pages = loader.load(
        PDF_PATH
    )

    print(
        f"Pages extracted: {len(pages)}"
    )

    # --------------------------------------------------
    # STEP 2: SPLIT INTO CHUNKS
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 2: TEXT CHUNKING")
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

    print("\nSample chunks:")

    for chunk in chunks[:3]:
        print("\n" + "-" * 70)
        print(
            f"Chunk ID: {chunk['chunk_id']}"
        )
        print(
            f"Page: {chunk['page']}"
        )
        print(
            f"Characters: {len(chunk['text'])}"
        )
        print(
            chunk["text"][:500]
        )

    # --------------------------------------------------
    # STEP 3: BUILD RETRIEVER
    # --------------------------------------------------

    print("\n" + "=" * 70)
    print("STEP 3: BUILD EMBEDDINGS + FAISS")
    print("=" * 70)

    retriever = RAGRetriever()

    retriever.build_index(
        chunks
    )

    print(
        "FAISS index built successfully."
    )

    # --------------------------------------------------
    # STEP 4: ASK QUESTIONS
    # --------------------------------------------------

    while True:

        question = input(
            "\nAsk a question about the PDF "
            "(type 'exit' to stop): "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        results = retriever.retrieve(
            question,
            top_k=5,
        )

        print("\n" + "=" * 70)
        print("RETRIEVED CHUNKS")
        print("=" * 70)

        for rank, result in enumerate(
            results,
            start=1,
        ):

            print("\n" + "-" * 70)

            print(
                f"Rank: {rank}"
            )

            print(
                f"Page: {result['page']}"
            )

            print(
                f"Chunk: {result['chunk_id']}"
            )

            print(
                f"Similarity: "
                f"{result['score']:.4f}"
            )

            print(
                result["text"][:1000]
            )


if __name__ == "__main__":
    main()