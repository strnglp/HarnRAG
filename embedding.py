import os
import logging

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
    SimpleKeywordTableIndex,
)
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.node_parser import SentenceSplitter
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

from pdf import MyPDFReader
logging.getLogger('bm25s').setLevel(logging.CRITICAL)
# scores are very low
RECIPROCAL_RANK = "reciprocal_rerank"  # apply reciprocal rank fusion
# I somewhat prefer this one
RELATIVE_SCORE = "relative_score"  # apply relative score fusion
# generally ok
DIST_BASED_SCORE = "dist_based_score"  # apply distance-based score fusion
# ok but not as good as relative
SIMPLE = "simple"  # simple re-ordering of results based on original scores


def get_query_engine(data_path, storage_path):
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
        vector_store = SimpleVectorStore()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents,
                                                storage_context=storage_context)
        storage_context.persist(persist_dir=storage_path)

    # Expecting a lot of high quality hits
    vector_retriever = index.as_retriever(similarity_top_k=20)
    # Give me your best 2 keyword hits
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=index.docstore, similarity_top_k=2)
    
    retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        # 6/4 worked well so far
        retriever_weights=[0.5, 0.5],
        similarity_top_k=20,
        num_queries=1,
        mode=RELATIVE_SCORE,
        use_async=True,
    )
    query_engine = RetrieverQueryEngine.from_args(retriever)
    # im not convinced keywords actually help...
#    query_engine = RetrieverQueryEngine.from_args(vector_retriever)
        
    return query_engine





