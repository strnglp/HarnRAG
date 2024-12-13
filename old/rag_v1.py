import os
import json
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores import SimpleVectorStore

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, RAKEKeywordTableIndex, StorageContext, load_index_from_storage
from llama_index.core.tools import QueryEngineTool, ToolMetadata

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama

from llama_index.core.agent import ReActAgent

# Be quiet please...
import sys
import logging
import warnings
logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL)
warnings.filterwarnings("ignore")
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"




folder_path = "./data/"
storage_path = "./storage/"

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

# bge-base embedding model TODO try others
Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5", device="cuda")
# ollama TODO try others
Settings.llm = Ollama(model="llama3", request_timeout=360.0)


os.makedirs(storage_path, exist_ok=True)

try:
    storage_context = StorageContext.from_defaults(persist_dir=storage_path)
    index = load_index_from_storage(storage_context)
    print("Index loaded from storage")
except:
    print("No index found. Creating...")
    pdf_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(".pdf")]
    reader = SimpleDirectoryReader(input_files=pdf_files)
    data = reader.load_data()
    index = VectorStoreIndex.from_documents(data)
    index.storage_context.persist(persist_dir=storage_path)

query_engine = index.as_query_engine(similarity_top_k=3)
while True:
    query_str = input(color_text("\n\nEnter your question (or 'q' to quit): ", "yellow"))
    if query_str.lower() == 'q':
        break
    response = query_engine.query(query_str)
    print(color_text("Response:", "underline"))
    print(color_text(response, "green"))
    print(color_text("Sources:", "underline"))
    for node in response.source_nodes:
        metadata = node.node.metadata
        print(color_text("Document:\t", "end"), end="")
        print(color_text(metadata.get("file_name"), "cyan"), end="")
        print(color_text(f" (page {metadata.get("page_label")})", "magenta"))



# agent mode
#query_engine_tool = QueryEngineTool(index.as_query_engine(),
#                       metadata=ToolMetadata(name="pdf_search",
#                                             description="Useful for searching across PDFs"))

#agent = ReActAgent.from_tools(
#    [query_engine_tool],
#    llm=Settings.llm,
#    max_iterations=50)

#response = agent.chat("Who are the Shek-Pvar and what pages cover that information?")
#print("Answer:" + response)
