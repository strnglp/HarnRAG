import os
import sys
import logging
import warnings
import textwrap
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

####
#### Logging, output & warning configuration
####
logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL)
warnings.filterwarnings("ignore")
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

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

####
#### LLM, Embedding, and Document setup
####
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5", device="cuda")
Settings.llm = Ollama(model="llama3", request_timeout=360.0)
folder_path = "./data/" # source data
storage_path = "./storage/" # computed data
os.makedirs(storage_path, exist_ok=True)

try:
    storage_context = StorageContext.from_defaults(persist_dir=storage_path)
    index = load_index_from_storage(storage_context)
    print("Index loaded from storage")
except Exception as e:
    print(f"No index found. Creating... ({e})")
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]
    reader = SimpleDirectoryReader(input_files=pdf_files)
    data = reader.load_data()
    index = VectorStoreIndex.from_documents(data)
    index.storage_context.persist(persist_dir=storage_path)

query_engine = index.as_query_engine(similarity_top_k=3)

####
#### User input loop for conversation
####
wrapper = textwrap.TextWrapper(width=80)
while True:
    print(color_text("\n\nEnter your question ('", "yellow"), end="")
    print(color_text(">", "white"), end="")
    print(color_text("' to follow up or '", "yellow"), end="")
    print(color_text("q", "red"), end="")
    print(color_text("' to quit): ", "yellow"), end="")
    query_str = input()
    if query_str.lower() == 'q':
        break
    response = query_engine.query(query_str)
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
    for node in response.source_nodes:
        metadata = node.node.metadata
        print(color_text("Document:\t", "end"), end="")
        print(color_text(metadata.get("file_name"), "cyan"), end="")
        print(color_text(f" (page {metadata.get("page_label")})", "magenta"))
