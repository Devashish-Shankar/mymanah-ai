import re


class TextSplitter:
    """
    Paragraph/sentence-aware text splitter.

    The splitter tries to preserve semantic boundaries while
    keeping chunks within a manageable character size.
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _clean_text(self, text: str) -> str:
        """
        Normalize whitespace while preserving paragraph boundaries.
        """

        text = text.replace("\r\n", "\n")
        text = text.replace("\r", "\n")

        # Remove excessive spaces.
        text = re.sub(r"[ \t]+", " ", text)

        # Remove excessive blank lines.
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    def _split_sentences(self, text: str) -> list[str]:
        """
        Basic sentence splitting.

        This intentionally avoids heavyweight NLP dependencies.
        """

        sentences = re.split(
            r"(?<=[.!?])\s+",
            text
        )

        return [
            sentence.strip()
            for sentence in sentences
            if sentence.strip()
        ]

    def split(self, pages: list[dict]) -> list[dict]:
        """
        Convert page-level text into semantic chunks.

        Each chunk contains:
        - chunk_id
        - page
        - text
        """

        chunks = []
        chunk_id = 0

        for page in pages:
            page_number = page["page"]
            text = self._clean_text(page["text"])

            if not text:
                continue

            paragraphs = re.split(
                r"\n\s*\n",
                text
            )

            current_chunk = ""

            for paragraph in paragraphs:
                paragraph = paragraph.strip()

                if not paragraph:
                    continue

                sentences = self._split_sentences(
                    paragraph
                )

                for sentence in sentences:

                    # If adding the sentence still fits.
                    candidate = (
                        f"{current_chunk} {sentence}".strip()
                    )

                    if len(candidate) <= self.chunk_size:
                        current_chunk = candidate
                        continue

                    # Save current chunk.
                    if current_chunk:
                        chunks.append(
                            {
                                "chunk_id": chunk_id,
                                "page": page_number,
                                "text": current_chunk,
                            }
                        )

                        chunk_id += 1

                    # Create overlap from previous chunk.
                    overlap_text = current_chunk[
                        -self.chunk_overlap:
                    ]

                    current_chunk = (
                        f"{overlap_text} {sentence}"
                    ).strip()

                    # If one sentence itself is very large,
                    # split it safely.
                    while len(current_chunk) > self.chunk_size:
                        piece = current_chunk[
                            :self.chunk_size
                        ].strip()

                        if piece:
                            chunks.append(
                                {
                                    "chunk_id": chunk_id,
                                    "page": page_number,
                                    "text": piece,
                                }
                            )

                            chunk_id += 1

                        current_chunk = current_chunk[
                            self.chunk_size
                            - self.chunk_overlap:
                        ].strip()

            # Save remaining text from this page.
            if current_chunk:
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "page": page_number,
                        "text": current_chunk,
                    }
                )

                chunk_id += 1

        return chunks