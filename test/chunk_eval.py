import os
import sys
import time
import logging
import pymupdf
import warnings
import nest_asyncio
from llama_index.core import (
    Document,
    SimpleDirectoryReader,
    VectorStoreIndex,
    ServiceContext,
    Settings,
)
from llama_index.core.evaluation import (
    DatasetGenerator,
    FaithfulnessEvaluator,
    RelevancyEvaluator
)
from llama_index.core.readers.base import BaseReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


nest_asyncio.apply()
logging.basicConfig(stream=sys.stdout, level=logging.CRITICAL)
warnings.filterwarnings("ignore") # pytorch can be spammy
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1" # amd ROCm support related

EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
LLM_MODEL = "llama3"


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

# Load Data
reader = SimpleDirectoryReader(input_dir="./data",
                               recursive=True,
                               file_extractor={".pdf": MyPDFReader()})
documents = reader.load_data()

# To evaluate for each chunk size, we will first generate a set of 40 questions from first 20 pages.
eval_documents = documents[:20]
data_generator = DatasetGenerator.from_documents(documents)
eval_questions = data_generator.generate_questions_from_nodes(num = 20)



# Define Faithfulness and Relevancy Evaluators which are based on GPT-4
faithfulness = FaithfulnessEvaluator()
relevancy = RelevancyEvaluator()

# Define function to calculate average response time, average faithfulness and average relevancy metrics for given chunk size
def evaluate_response_time_and_accuracy(chunk_size, overlap):
    Settings.chunk_size = chunk_size
    Settings.overlap = overlap
    total_response_time = 0
    total_faithfulness = 0
    total_relevancy = 0

    # create vector index
    llm = Settings.llm

    vector_index = VectorStoreIndex.from_documents(
        eval_documents,
    )

    query_engine = vector_index.as_query_engine()
    num_questions = len(eval_questions)

    for question in eval_questions:
        start_time = time.time()
        response_vector = query_engine.query(question)
        elapsed_time = time.time() - start_time
        
        faithfulness_result = faithfulness.evaluate_response(
            response=response_vector
        ).passing
        
        relevancy_result = relevancy.evaluate_response(
            query=question, response=response_vector
        ).passing

        total_response_time += elapsed_time
        total_faithfulness += faithfulness_result
        total_relevancy += relevancy_result

    average_response_time = total_response_time / num_questions
    average_faithfulness = total_faithfulness / num_questions
    average_relevancy = total_relevancy / num_questions

    return average_response_time, average_faithfulness, average_relevancy

# Iterate over different chunk sizes to evaluate the metrics to help fix the chunk size.
for overlap in [10, 20, 30, 40, 50]:
    for chunk_size in [128, 256, 512, 1024, 2048]:
        avg_time, avg_faithfulness, avg_relevancy = evaluate_response_time_and_accuracy(chunk_size, overlap)
        print(f"Chunk size {chunk_size} Overlap {overlap} - Average Response time: {avg_time:.2f}s, Average Faithfulness: {avg_faithfulness:.2f}, Average Relevancy: {avg_relevancy:.2f}")
