import os
import sys
import logging
import warnings
import textwrap
import pymupdf
from llama_index.core import (
    Document, VectorStoreIndex, Settings, StorageContext, load_index_from_storage, SimpleDirectoryReader, PromptTemplate
)
from llama_index.core.readers.base import BaseReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# seems like a good optimization to include keywords
# https://docs.llamaindex.ai/en/stable/examples/query_engine/CustomRetrievers/ 
################# AI MODEL SETTINGS ###################
# This project can use various embedding models for
# encoding semantic meaning of documents into vectors
# https://huggingface.co/BAAI/bge-base-en-v1.5

# Baseline best performance I've seen, fails on
# ritual invocations
EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"

# Very wordy and missed lists of things like spells
#EMBEDDING_MODEL = "dunzhang/stella_en_1.5B_v5"

# A close 2nd place but bad when using the provided instructions
#EMBEDDING_MODEL = "intfloat/e5-large-v2"

# This project uses Ollama models for summarization of 
# answers, https://ollama.com/search
LLM_MODEL = "llama3"
#######################################################

####
#### Logging, output & warning configuration
####
logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL)
warnings.filterwarnings("ignore") # pytorch can be spammy
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1" # amd ROCm support related

def color_text(text, color):
    """Colors text for terminal output.

    Args:
        text: The text to color.
        color: The color name ('cyan', 'magenta', or other supported colors).

    Returns:
        The colored text string, or the original text if the color is not supported.
    """

    color_codes = {
        "cyan": "\033[96m",
        "magenta": "\033[95m",
        "blue": "\033[94m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "red": "\033[91m",
        "bold": "\033[1m",
        "underline": "\033[4m",
        "end": "\033[0m",  # Reset to default color
    }
    return f"{color_codes.get(color, '')}{text}{color_codes['end']}"

####
#### Conversation & context related functions
####
conversation_history = []
def get_follow_up_context():
    """Function to retrieve the follow-up context (only the last Q&A chain)

    Returns:
        All previous context for questions that begin with '>' up to the
        first question that doesn't (original question).
        In "Q: <question> A: <answer>" form.
    """
    global conversation_history
    # Check the conversation history for follow-ups
    context = ""
    # Start from the most recent Q&A and go backward while questions have the ">" prefix
    for entry in reversed(conversation_history):
        if entry['question'].startswith(">"):
            context = f"Q: {entry['question'][1:].strip()}\nA: {entry['answer']}\n" + context
        else:
            # Stop accumulating context when a non-follow-up question is found
            context = f"Q: {entry['question'][1:].strip()}\nA: {entry['answer']}\n" + context
            break
    return context

def query_with_follow_up(question, query_engine):
    """Combines prior context with current question as necessary before querying

    Args:
        question: The user query ('>' prefix will collect prior context)
        query_engine: llama-index supports multiple types of query engines
    Returns:
        LLM generated response to the question (and optional context)
    """

    global conversation_history

    if question.startswith(">"):
        # It's a follow-up, so add prior Q&A to context
        print(color_text("Adding prior Q&A to context.", "blue"))
        context = get_follow_up_context()
        question_without_prefix = question[1:].strip()
        question_with_context = f"{context}\nQ: {question_without_prefix}"
    else:
        # It's an unrelated question, no context from prior Q&As
        question_with_context = question

    response = query_engine.query(question_with_context)

    # Save the question and answer to conversation history
    conversation_history.append({
        "question": question,
        "answer": response.response,
    })

    return response

class MyPDFReader(BaseReader):
    
    # Method to extract text with clipping and textpage creation
    def extract_text(self, page):
        page_rect = page.rect
        clip_rect = pymupdf.Rect(page_rect.x0, page_rect.y0, page_rect.x1, page_rect.y1 - 70)
        text_page = page.get_textpage(clip=clip_rect)
        return page.get_text("text", textpage=text_page, clip=clip_rect)
    
    def load_data(self, file, extra_info=None):
        with pymupdf.open(file) as doc:
            documents = []
            for page_number, page in enumerate(doc, start=1):
                text = self.extract_text(page)
                doc_metadata = extra_info or {}
                doc_metadata.update({"page_number": page_number})
                documents.append(Document(text=text, extra_info=doc_metadata))
        return documents

####
#### LLM, Embedding, and Document setup
####
Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL, 
    device="cuda", 
    max_length=512,  # Model's max token length
    trust_remote_code=True
)

# context window size doesn't seem to perfectly solve the
# accuracy problem as the corpus grows, may need a router
Settings.llm = Ollama(
    model=LLM_MODEL,
    temperature=0,
    request_timeout=60.0
)

data_path = "./data/" # source data
storage_path = "./storage_" + EMBEDDING_MODEL.split("/")[1] # computed data
os.makedirs(storage_path, exist_ok=True)

try:
    print("Loading index from storage...")
    storage_context = StorageContext.from_defaults(persist_dir=storage_path)
    index = load_index_from_storage(storage_context)
except Exception as e:
    print("No index found. Creating...")
    reader = SimpleDirectoryReader(input_dir=data_path,
                                   recursive=True,
                                   file_extractor={".pdf": MyPDFReader()})
    documents = reader.load_data()

    index = VectorStoreIndex.from_documents(documents)
    index.storage_context.persist(persist_dir=storage_path)


# Query configuration
query_engine = index.as_query_engine(similarity_top_k=3)

####
#### User input loop for conversation
####
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
    # different loaders can change the format of metadata
    #print(color_text(response.source_nodes, "red"))
    for node in response.source_nodes:
        metadata = node.node.metadata
        print(color_text("Document:\t", "end"), end="")
        print(color_text(metadata.get("file_path", "").split("/")[-1], "cyan"), end="")
        print(color_text(f" (page {metadata.get("page_number")})", "magenta"))
