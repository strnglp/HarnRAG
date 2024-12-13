#!/bin/bash

# Check for required arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 /path/to/inputdir /path/to/outputdir"
    exit 1
fi

input_dir="$1"
output_dir="$2"
working_dir="$output_dir/working"
mkdir -p "$working_dir"

# Find PDF files
mapfile -d '' pdf_files < <(find "$input_dir" -type f -name "*.pdf" -print0)

# Loop through the pdf files
for pdf in "${pdf_files[@]}"; do

    pdf_base_filename=$(basename "$pdf" .pdf)

    # Check if already searchable
    line_count=$(pdftotext "$pdf" - | wc -l)
    if [ "$line_count" -gt 50 ]; then
        echo "$pdf: Skipping, already searchable. Copying to output directory."
        cp "$pdf" "$output_dir/"
        continue
    fi

    echo "Processing: $pdf..."

    # Turn PDF into a series of PNGs
    pdftoppm -png -r 300 "$pdf" "$working_dir/page"

    # OCR each image and combine into PDFs
    for img_file in "$working_dir/page-"*.png; do
        img_base_filename=$(basename "$img_file" .png)
        tesseract-ocr "$img_file" "$working_dir/ocr_$img_base_filename" -l eng+deu+fra+lat pdf
    done

    # Combine OCR'd PDF pages into a single PDF
    pdftk "$working_dir"/ocr_page-*.pdf cat output "$working_dir/$pdf_base_filename.pdf"

    # Clean up PNGs and intermediate files
    rm "$working_dir/page-"*.png "$working_dir"/ocr_page-*.pdf

    # Move final output to the output directory
    mv "$working_dir/$pdf_base_filename.pdf" "$output_dir/"
done
