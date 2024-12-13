from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core import QueryBundle
from llama_index.core.schema import NodeWithScore
from llama_index.core import get_response_synthesizer
from typing import List

class CustomRetriever(BaseRetriever):
    """Custom retriever that performs both semantic search and hybrid search."""

    def __init__(
        self,
        vector_retriever: VectorIndexRetriever,
        keyword_retriever: KeywordTableSimpleRetriever,
        mode: str = "AND",
        min_relevancy: float = 0.5
    ) -> None:
        """Init params."""

        self._vector_retriever = vector_retriever
        self._keyword_retriever = keyword_retriever
        if mode not in ("AND", "OR"):
            raise ValueError("Invalid mode.")
        self._mode = mode
        self._min_relevancy = min_relevancy
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        """Retrieve nodes given query."""

        vector_nodes = self._vector_retriever.retrieve(query_bundle)
        keyword_nodes = self._keyword_retriever.retrieve(query_bundle)

        vector_ids = {n.node.node_id for n in vector_nodes if n.score > self._min_relevancy}
        keyword_ids = {n.node.node_id for n in keyword_nodes}

        combined_dict = {n.node.node_id: n for n in vector_nodes}
        combined_dict.update({n.node.node_id: n for n in keyword_nodes})

        if self._mode == "AND":
            retrieve_ids = vector_ids.intersection(keyword_ids)
        else:
            retrieve_ids = vector_ids.union(keyword_ids)

        retrieve_nodes = [combined_dict[rid] for rid in retrieve_ids]
        return retrieve_nodes

def get_vector_retriever(vector_index, top_k):
    return VectorIndexRetriever(index=vector_index, similarity_top_k=top_k)

def get_keyword_retriever(keyword_index, top_k):
    return KeywordTableSimpleRetriever(index=keyword_index, similarity_top_k=top_k)

def get_custom_query_engine(vector_index,
                            keyword_index,
                            top_k,
                            mode,
                            min_relevancy):
    custom_retriever = CustomRetriever(get_vector_retriever(vector_index, top_k),
                                       get_keyword_retriever(keyword_index), mode, min_relevancy)
    return RetrieverQueryEngine(
        retriever=custom_retriever,
        response_synthesizer=get_response_synthesizer())

def get_vector_query_engine(vector_index, top_k):
    return RetrieverQueryEngine(
        retriever=get_vector_retriever(vector_index, top_k),
        response_synthesizer=get_response_synthesizer())

def get_keyword_query_engine(keyword_index, top_k):
    return RetrieverQueryEngine(
        retriever=get_keyword_retriever(keyword_index, top_k),
        response_synthesizer=get_response_synthesizer())

def get_keyword_vector_query_engine(vector_index,
                                    keyword_index,
                                    top_k: int = 10,
                                    keyword_filter_top_k: int = 20,
                                    min_relevancy: float = 0.5):
    custom_retriever = KeywordVectorRetriever(
        get_vector_retriever(vector_index, top_k),
        get_keyword_retriever(keyword_index, keyword_filter_top_k),
        min_relevancy)
    return RetrieverQueryEngine(
        retriever=custom_retriever,
        response_synthesizer=get_response_synthesizer())


def get_hybrid_query_engine(vector_index, top_k):
    return RetrieverQueryEngine(
        retriever=get_vector_retriever(vector_index, top_k),
        response_synthesizer=get_response_synthesizer())
