import json
import re

def enrich_markdown_with_metadata(md_text, metadata_json):
    try:
        # Load and parse the JSON file
        with open(metadata_json, "r") as f:
            metadata = json.load(f)
        
        # Focus on the table_of_contents key
        toc = metadata.get("table_of_contents", [])
        if not isinstance(toc, list):
            raise ValueError("'table_of_contents' must be a list.")
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Error loading metadata: {e}")
        return md_text  # Return original markdown if metadata is invalid

    enriched_md = []
    for item in toc:
        # Ensure required fields exist
        if not all(key in item for key in ["title", "heading_level", "page_id"]):
            print(f"Skipping item due to missing fields: {item}")
            continue
        
        # Build metadata comment
        meta_comment = f"<!-- title: {item['title']}, heading_level: {item['heading_level']}, page_id: {item['page_id']} -->"
        enriched_md.append(meta_comment)
        
        # Adjust heading in Markdown
        heading_marker = "#" * item["heading_level"]
        enriched_md.append(f"{heading_marker} {item['title']}")
    
    # Combine enriched content with existing markdown text
    enriched_md.append("\n" + md_text)
    return "\n".join(enriched_md)

with open("/home/jv/AI/tmp/CG 4301 - HarnMaster Religion/CG 4301 - HarnMaster Religion.md", "r") as f:
    markdown = f.read()

enriched_markdown = enrich_markdown_with_metadata(markdown, "/home/jv/AI/tmp/CG 4301 - HarnMaster Religion/CG 4301 - HarnMaster Religion_meta.json")
with open("/home/jv/AI/tmp/CG 4301 - HarnMaster Religion/CG 4301 - HarnMaster Religion - enriched.md", "w") as f:
    f.write(enriched_markdown)
