from app.rag.loader import PDFLoader


PDF_PATH = "data/sample.pdf"


loader = PDFLoader()

pages = loader.load(PDF_PATH)

print("=" * 70)
print("PDF LOADER TEST")
print("=" * 70)

print(f"Total pages with extracted text: {len(pages)}")

for page in pages[:3]:
    print("\n" + "-" * 70)
    print(f"PAGE: {page['page']}")
    print("-" * 70)
    print(page["text"][:1000])