import sys
import pymupdf
import re

impossible_pattern = r'\b(?:[^aeiou\s]{4,}|[aeiou]{4,})\b'

def get_filtered_text_from_page(page, min_words_per_page=200, top_n=3,
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
    
                            
def get_filtered_text_from_pdf(pdf_path, min_words_per_page=200, top_n=3,
                               min_size=12, footer_cutoff=0, debug=False):


    filtered_text = []
    with pymupdf.open(pdf_path) as doc:
        for page_num in range(len(doc)):

            page = doc[page_num]
            clip_rect = pymupdf.Rect(page.rect.x0, page.rect.y0, page.rect.x1, page.rect.y1 - footer_cutoff)
            tp = page.get_textpage(clip=clip_rect)
            page_filtered = get_filtered_text_from_page(tp, min_words_per_page,
                                                        top_n, min_size,
                                                        debug)
            # specialization, sometimes the title of the page is unreadable
            # due to font encoding / missing ToUnicode tables so make the
            # assumption that if it is the first item in this list, it was
            # most likely the page title and lets replace it with pymupdf's
            # best take on what that text was. This is a side effect of
            # TextPage providing low level interfaces that don't abstact
            # all of the work Page.get_text will do
            if page_filtered and re.match(r'�{3,}', page_filtered[0]):
                page_filtered[0] = page.get_text().splitlines()[0]
            filtered_text.append(page_filtered)
    if debug:
        for text in filtered_text:
            print(text)
    return filtered_text

pdf_path = sys.argv[1]
largest_texts = get_filtered_text_from_pdf(pdf_path, min_words_per_page=100,
                                           top_n=2, min_size=12,
                                           footer_cutoff=70, debug=True)




