import json
import re
import os
import spacy
import nltk
from nltk.corpus import wordnet
import numpy as np
from typing import List, Optional
from pydantic import Field, BaseModel

from llama_index.core import (
    Settings,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
    QueryBundle
)
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor, LLMRerank
from llama_index.core.postprocessor.types import BaseNodePostprocessor
from llama_index.core.schema import NodeWithScore
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    KeywordTableSimpleRetriever,
)

from pdf import MyPDFReader
nltk.download('wordnet')
nlp = spacy.load("en_core_web_sm")
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
        

        for doc in documents:
           combined_embedding = create_weighted_document_embedding(doc)
           doc.embedding = combined_embedding


        vector_store = SimpleVectorStore()
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents,
                                                storage_context=storage_context)
        storage_context.persist(persist_dir=storage_path)


    vector_retriever = index.as_retriever(similarity_top_k=10)
    query_engine = RetrieverQueryEngine.from_args(vector_retriever,
                                                  response_mode='compact',
                                                  node_postprocessors=[
                                                      #LLMRerank(choice_batch_size=20, top_n=5)
                                                      KWBoostPostprocessor()
                                                  ]
                                                  )
        
    return query_engine



class KWBoostPostprocessor(BaseNodePostprocessor, BaseModel):
    metadata_weight: float = 1.0
    content_weight: float = 1.0

    def __init__(self, metadata_weight=1, content_weight=1):
        super().__init__()
        self.metadata_weight = metadata_weight
        self.content_weight = content_weight

    def _postprocess_nodes(self, nodes, query):
        query_terms = self._extract_terms(query.query_str)
        for node in nodes:
            keywords = [node.metadata.get("book", "")]
            keywords += node.metadata.get("keywords", "").split(",")
            normalized_keywords = self._normalize_keywords(keywords)
            matching_terms = query_terms.intersection(normalized_keywords)
            match_count = len(matching_terms)
            if match_count > 0:
                boost_factor = 1.0 + (0.3 * match_count)
                node.score *= boost_factor

        # If we're confident we found high quality matches, just return them
        filtered_nodes = [node for node in nodes if node.score > 0.7]
        if len(filtered_nodes) > 0:
            return filtered_nodes

        # otherwise return the full modified set
        return nodes

    def _extract_terms(self, query_str):
        return {token.lemma_.lower() for token in nlp(query_str) if token.pos_ == "PROPN" and token.is_alpha}

    def _normalize_keywords(self, keywords):
        return {token.lemma_.lower() for keyword in keywords for token in nlp(keyword.lower().strip()) if token.pos_ == "PROPN" and token.is_alpha}
    
    
def extract_keywords(metadata_str):
    try:
        metadata_str = re.sub(r"page: \d+,", "", metadata_str)
        book = re.search(r"book:\s*([^\n]+)", metadata_str)
        book = book.group(1) if book else ""
        book = re.sub(r"-", " ", book)
        book = re.sub(r"\d+", "", book)
        book = book.strip()
        keywords = re.findall(r"keywords:\s*([^,]+(?:,\s*[^,]+)*)", metadata_str)
        keywords = [keyword.strip() for keyword in keywords[0].split(',') if keyword.strip()] if keywords else []
        keywords.append(book)
        return sorted(keywords)
    except Exception as e:
        print(f"Error processing metadata: {e}")
        return []


def is_common_noun(word):
    synsets = wordnet.synsets(word.lower())
    return any(synset.pos() == 'n' for synset in synsets) 

def boost_terms(terms, weight=0.5):
    term_embedding = Settings.embed_model.get_text_embedding("")
    boosted_embedding = np.zeros_like(term_embedding)

    for term in terms:
        term_embedding = Settings.embed_model.get_text_embedding(term)
        boosted_embedding += weight * np.array(term_embedding)

    return boosted_embedding


def create_weighted_document_embedding(doc, metadata_weight=1, content_weight=1):
    content = doc.text
    metadata = doc.get_metadata_str()

    keywords = extract_keywords(metadata)
    common_nouns = list(filter(is_common_noun, keywords))
    proper_nouns = list(filter(lambda word: not is_common_noun(word), keywords))

    # Boost embeddings for terms
    proper_noun_boost = boost_terms(proper_nouns, weight=1.5) / max(1, len(proper_nouns))
    # Don't boost common nouns, it pollutes the analysis by over indexing on common terms across
    # concepts that would otherwise be narrowed by proper nouns, such as "Peoni Spells"
    #common_noun_boost = boost_terms(common_nouns, weight=1.25) / max(1, len(common_nouns))
    common_noun_boost = 0

    # Get content embedding
    content_embedding = np.array(Settings.embed_model.get_text_embedding(content))

    # Combine embeddings with weights
    combined_embedding = (
        content_weight * content_embedding +
        metadata_weight * (proper_noun_boost + common_noun_boost)
    )

    # Normalize the combined embedding (optional, depending on use case)
    combined_embedding = combined_embedding / np.linalg.norm(combined_embedding)

    return combined_embedding
