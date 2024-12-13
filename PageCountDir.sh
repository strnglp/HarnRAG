#!/bin/bash
input_dir="$1"
total_pages=0

# Loop over each PDF file in the directory
for pdf in "$input_dir"/*.pdf; do
    # Get the page count using pdfinfo
    page_count=$(pdfinfo "$pdf" | grep "Pages:" | awk '{print $2}')
    
    # Add the page count to the total
    total_pages=$((total_pages + page_count))
    
    echo "$pdf has $page_count pages"
done

# Print the running total
echo "Total pages: $total_pages"
