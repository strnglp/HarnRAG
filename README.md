# A RAG experiment
I wrote this to learn more about AI and HarnMaster the TTRPG 🤓

## Overview
This is a relatively basic RAG implementation using llama-index which
I've specialized around a specific form of PDF content.

## Files
`rag.py`
This file has seen many revisions (documented in `old/`) as I
experimented. Today it is the entry point and where I define my LLM
configuration and handle user input and prompts.

`embedding.py`
Contains most of the AI stuff where it constructs a somewhat
specialized retriever for the query engine that combines BM25 keyword
search with vector search, ordered by a relative score. I'm still
experimenting with the values, most frequently the  `weights` and `top_k`.

`history.py`
This is my low-budget attempt at managing to answer follow up
questions with a relatively small context window. I can't have the LLM
continuously build up context with every question as I don't have
nearly enough VRAM for that. So I explicitly include follow up
questions with the '>' character. Works well enough for me.

`pdf.py`
This is my "specialized" PDF reader for HarnWorld / HarnMaster PDF content.

Some of the specailizations are...
1. Scan the page for the largest fonts and consider those context
clues for the LLM.
2. Detect patterns that are related to "Spells" in HarnMaster and add
a hint as a context clue for the LLM.
3. Expect that not all content was OCRed correctly so ignore gibberish
or unprintables.
4. Use a clipping rectangle to ignore the footer of the page which is
the same unrelated content (copyright) over and over.
5. Ignore pages that are mostly images.

`color.py`
Nothing interesting here, just a function for using terminal colors.

`test/`
Some test programs I wrote or downloaded as I experimented with
post-processing.

`MakeSearchablePDF.sh` & `PageCountDir.sh`
Just some simple scripts I wrote to OCR the PDF scans into usable PDFs
with embeded textual representation.

## Requirements
`requirements.txt` is a `pip freeze` from today... it's probably chock
full of cruft that I ultimately didn't use or need. :)

`pip install requirements.txt` if you're brave. Always best to use a
virtual environment.

## Usage
Create a `data/` directory and fill it with searchable PDFs.

`ollama pull llama3.1`, or whatever LLM you have configured to act as
an agent.

`ollama serve`, you might want to pipe to /dev/null or run this in the background somewhere
as it will flood the terminal.

`python rag.py` will start building the index (saved to
`storage_<embedding-model-name>/`) or load an index if one
is detected. That means you should delete your `storage_*/` directory
if you want to reindex.