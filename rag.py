import os
import sys
import logging
import warnings

from llama_index.core import Settings
from llama_index.core.prompts import PromptTemplate
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

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
#logging.basicConfig(level=logging.CRITICAL)
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
