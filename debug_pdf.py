import pdfplumber

pdf_path = "data/ug_bulletin_2025-2026.pdf"

with pdfplumber.open(pdf_path) as pdf:
    print(f"Total pages: {len(pdf.pages)}")
    for page_num in [1, 2, 3, 10, 50]:
        page = pdf.pages[page_num - 1]
        text = page.extract_text() or ""
        words = page.extract_words()
        print(f"\n--- Page {page_num} ---")
        print(f"  extract_text() chars : {len(text)}")
        print(f"  extract_words() count: {len(words)}")
        print(f"  text preview         : {repr(text[:200])}")
