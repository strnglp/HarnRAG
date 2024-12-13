import os
import re
import pymupdf
from llama_index.core.readers.base import BaseReader
from llama_index.core import (
    Document,
)

class MyPDFReader(BaseReader):
    def load_data(self, file, extra_info=None):
        with pymupdf.open(file) as doc:
            documents = []
            for page_number, page in enumerate(doc, start=1):
                page_content = extract_text(page)
                doc_metadata = {
                    "file": os.path.splitext(os.path.basename(file))[0],
                    "page": page_number,
                    "related": page_content["related"]}
                documents.append(Document(text=page_content["text"],
                                          metadata=doc_metadata))
        return documents

def extract_text(page):
    page_rect = page.rect
    clip_rect = pymupdf.Rect(page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1 - 70)
    text_page = page.get_textpage(clip=clip_rect)
    text = page.get_text("text", textpage=text_page)
    related = get_filtered_text_from_page(text_page,
                                          min_words_per_page=50,
                                          top_n=2) or []
    # specalization...

    if related and re.match(r'�{3,}', related[0]):
        related[0] = page.get_text().splitlines()[0]

    spell_term_pattern = [r"Time:", r"Range:", r"Duration:"]
    roman_numeral_pattern = r"\((?:[IVXLCDM]+)\)"
    if (all(re.search(pattern, text, re.IGNORECASE) for pattern in spell_term_pattern)
        or re.search(roman_numeral_pattern, text)):
        related.append("SPELL")

    return {"text": text, "related": ",".join(related) if related else ""}


impossible_pattern = r'\b(?:[^aeiou\s]{4,}|[aeiou]{4,})\b'
def get_filtered_text_from_page(page, min_words_per_page=100, top_n=3,
                                min_size=12, debug=False):
    words = page.extractText().split()

    # Skip probable title/copyright pages
    if len(words) < min_words_per_page:
        return
    
    text_with_font_size = []
    blocks = page.extractDICT()["blocks"]

    for block in blocks:
        if "lines" in block:
            for line in block["lines"]:
                # Reconstruct the text for this line by combining spans
                line_text = ""
                line_font_size = None
                
                for span in line["spans"]:
                    # Reconstruct the line text by combining spans
                    line_text += span["text"]
                    if line_font_size is None:
                        line_font_size = span["size"]

                text = re.sub(r"[^\D\s]", "", line_text, flags=re.UNICODE)  # Allow Unicode word characters but not digits
                text = text.strip()
                if len(text) > 3 and len(text) < 40 and not re.search(impossible_pattern, text, re.IGNORECASE):
                    text_with_font_size.append((line_font_size, text))

    # Get the top 'n' unique font sizes
    unique_sizes = sorted(set([size for size, text in text_with_font_size]), reverse=True)[:top_n]
    # Filter the text that matches one of the top 'n' sizes
    filtered_text = [t for size, t in text_with_font_size if size in unique_sizes and size > min_size]
    
    return filtered_text
