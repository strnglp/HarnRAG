import os
import sys
import logging
import warnings
import textwrap
from tabulate import tabulate
from collections import defaultdict
from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from color import color_text
from embedding import get_query_engine
from history import query_with_follow_up
################# AI MODEL SETTINGS ###################
# This project can use various embedding models for
# encoding semantic meaning of documents into vectors
# https://huggingface.co/BAAI/bge-base-en-v1.5
# Baseline best performance I've seen
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
# This project uses Ollama for summarization of answers
# https://ollama.com/search
LLM_MODEL = "llama3.1"
Settings.chunk_size = 512
Settings.overlap = 50
Settings.context_window = 20000
Settings.num_output = 2048
Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL, 
    device="cuda", 
    max_length=512,
    trust_remote_code=True
)
Settings.llm = Ollama(
    model=LLM_MODEL,
    temperature=0,
    request_timeout=60.0,
    context_window=Settings.context_window,
    num_output=Settings.num_output,
)
#######################################################
####
#### Logging, output & warning configuration
####
logging.basicConfig(level=logging.CRITICAL)
warnings.filterwarnings("ignore") # pytorch can be spammy
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1" # amd ROCm support related
###
### Load or create embeddings
###
data_path = "./data/" # source data
storage_path = "./storage_" + EMBEDDING_MODEL.split("/")[1] # computed data
#vector_index, keyword_index = get_embeddings(data_path, storage_path)
####
#### User input loop for conversation
####
# TODO: swap in a custom prompt when "lists" or quantities (all) are requested
custom_qa_prompt_str = """
Context information is below.
---------------------
{context_str}
Note: Spells in the context of Gods, Clerics, or Religion are actually synonymous with Ritual Invocations.
---------------------
After a comprehensive analysis of the context information, answer the query succinctly, summarizing as necessary.
Do not reference the context or the context information.
Query: {query_str}
Answer:
"""
custom_qa_prompt = PromptTemplate(custom_qa_prompt_str)
query_engine = get_query_engine(data_path, storage_path)
query_engine.update_prompts({"response_synthesizer:text_qa_template": custom_qa_prompt})

wrapper = textwrap.TextWrapper(width=80)
while True:
    print(color_text("\n\nEnter your question ('", "yellow"), end="")
    print(color_text(">", "blue"), end="")
    print(color_text("' to follow up or '", "yellow"), end="")
    print(color_text("q", "red"), end="")
    print(color_text("' to quit): ", "yellow"), end="")
    query_str = input()
    if not query_str:
        continue
    if query_str.lower() == 'q':
        break
    response = query_with_follow_up(query_str, query_engine)
    print(color_text("\nResponse:", "underline"))
    response_text = '\n'.join([
        '\n'.join(
            textwrap.wrap(
                line, 90, break_long_words=False, replace_whitespace=False
            )
        ) if line.strip() != '' else ''  # Keep blank lines intact
        for line in response.response.splitlines()
    ])
    print(color_text(response_text, "green"))
    print(color_text("\nSources:", "underline"))

    sorted_nodes = sorted(response.source_nodes, key=lambda node: node.score, reverse=True)
    book_pages = defaultdict(set)
    for node in sorted_nodes:
        metadata = node.node.metadata
        book_pages[metadata["file"]].add(metadata["page"])


    list_of_pairs = [
        [
            color_text("📘"+book, "blue"),
            color_text("📑"+", ".join(str(page) for page in pages), "magenta")
        ]
        for book, pages in book_pages.items()
    ]

    table = tabulate(list_of_pairs,
                     tablefmt="fancy_grid",
                     maxcolwidths=[60, 30],
                     colalign=("left", "left"))

    yellow = "\033[93m"
    reset = "\033[0m"
    table = table.replace(reset, yellow)
    print(color_text(table, "yellow"))
