import fitz


class PDFLoader:
    """
    Loads a PDF document and extracts text page by page.
    """

    def load(self, file_path: str) -> list[dict]:
        document = fitz.open(file_path)

        pages = []

        for page_number, page in enumerate(document, start=1):
            text = page.get_text("text").strip()

            if text:
                pages.append(
                    {
                        "page": page_number,
                        "text": text,
                    }
                )

        document.close()

        return pages