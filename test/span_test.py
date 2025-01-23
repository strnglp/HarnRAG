import fitz  # PyMuPDF
import argparse
def highlight_first_and_last_span(pdf_path, output_path):
    # Open the PDF file
    doc = fitz.open(pdf_path)

    # Iterate over all pages in the document
    for page_num in range(doc.page_count):
        page = doc[page_num]

        # Extract text spans from the page
        text_spans = page.get_text("dict")["blocks"]

        # Extract bounding boxes of text spans
        bounding_boxes = []
        for block in text_spans:
            if block["type"] == 0:  # Text block
                for line in block["lines"]:
                    x0, y0, x1, y1 = line["bbox"]  # Bounding box of the span
                    bounding_boxes.append((x0, y0, x1, y1))

        if not bounding_boxes:
            print(f"No text spans found on page {page_num + 1}.")
            continue

        # Get the first and last bounding boxes
        first_box = bounding_boxes[0]
        last_box = bounding_boxes[-1]

        # Highlight the first and last text span on this page
        rect_first = fitz.Rect(first_box)
        rect_last = fitz.Rect(last_box)

        page.draw_rect(rect_first, color=(0, 1, 0), width=2)  # Green for the first span
        page.draw_rect(rect_last, color=(1, 0, 0), width=2)  # Red for the last span

    # Save the output PDF with highlighted text spans
    doc.save(output_path)

def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Identify and cluster text blocks in a PDF")
    parser.add_argument("pdf_path", help="Path to the input PDF file")
    parser.add_argument("output_path", help="Path to save the output PDF with bounding boxes")
    args = parser.parse_args()

    # Input and output paths
    pdf_path = args.pdf_path
    output_path = args.output_path

    highlight_first_and_last_span(pdf_path, output_path)

if __name__ == "__main__":
    main()
