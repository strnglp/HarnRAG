import os
import re
import pymupdf
from llama_index.core.readers.base import BaseReader
from llama_index.core import (
    Document,
)
from multi_column import column_boxes

class MyPDFReader(BaseReader):
    def load_data(self, file, extra_info=None):
        print(f"Processing {file}")
        with pymupdf.open(file) as doc:
            book = os.path.splitext(os.path.basename(file))[0]
            documents = []
            for page_number, page in enumerate(doc, start=1):
                page_content = extract_text(page, book)
                doc_metadata = {
                    "book": book,
                    "page": page_number}
                if page_content['related']:
                    doc_metadata['keywords'] = page_content['related']

                doc = Document(text=page_content['text'], metadata=doc_metadata)
                documents.append(doc)
        return documents

def extract_text(page, book):

    # use column_boxes to pull out higher quality and sequenced content
    # than get_text would...
    text_boxes = column_boxes(page, 70, 70)

    # ignore header/footer
    page_rect = page.rect
    clip_rect_with_header = pymupdf.Rect(page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1 - 70)
    clip_rect_without_header = pymupdf.Rect(page_rect.x0, page_rect.y0 + 70, page_rect.x1, page_rect.y1 - 70)



    # we don't use the header when extracting content
    text_page = page.get_textpage(clip=clip_rect_without_header)
    text = ""
    for tb in text_boxes:
        text += page.get_text("text", clip=tb,
                              flags=pymupdf.TEXTFLAGS_TEXT|pymupdf.TEXT_DEHYPHENATE|pymupdf.TEXT_INHIBIT_SPACES,
                              textpage=text_page,
                              sort=True) + "\n"

    # we do want the header though when extracting keywords
    text_page = page.get_textpage(clip=clip_rect_with_header)
    related = get_filtered_text_from_page(text_page,
                                          min_words_per_page=50,
                                          top_n=2) or []
    # specalization...

    if related and re.match(r'�{3,}', related[0]):
        related[0] = page.get_text().splitlines()[0]

    spell_term_pattern = [r"Time:", r"Range:", r"Duration:"]
    roman_numeral_pattern = r"\((?:[IVXLCDM]+)\)"
    roman_numerals = len(re.findall(roman_numeral_pattern, text, re.IGNORECASE))
    matched_spell = all(re.search(pattern, text, re.IGNORECASE) for pattern in spell_term_pattern)
    if (matched_spell or roman_numerals > 0):
        if roman_numerals > 1:
            related.append("Spell List")
        else:
            related.append("Spell")
        # In Harn divine magics are ritual invocations, but let's also include Spell
        # for completeness as not many people would make the distinction
        if "Religion" in book:
            if roman_numerals > 1:
                related.append("Ritual Invocation List")
            else:
                related.append("Ritual Invocation")

    return {"text": text, "related": ", ".join(related) if related else ""}


impossible_pattern = r'\b(?:[^aeiou\s]{4,}|[aeiou]{4,})\b'
def get_filtered_text_from_page(page, min_words_per_page=100, top_n=3, min_size=13):
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
                        line_font_size = round(span["size"])

                text = re.sub(r"[^\D\s]", "", line_text, flags=re.UNICODE)  # Allow Unicode word characters but not digits
                text = text.strip()
                if len(text) > 3 and len(text) < 40 and not re.search(impossible_pattern, text, re.IGNORECASE):
                    text_with_font_size.append((line_font_size, text))

    # Get the top 'n' unique font sizes
    unique_sizes = sorted(set([size for size, text in text_with_font_size]), reverse=True)[:top_n]
    # Filter the text that matches one of the top 'n' sizes
    filtered_text = [t for size, t in text_with_font_size if size in unique_sizes and size >= min_size]
    
    return filtered_text
