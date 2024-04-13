import logging

from embeddings.Embeddings import Embeddings
from embeddings.LocalHFEmbeddings import LocalEmbeddings
from ingestion.chunking.Chunk import ChildChunk, ParentChunk
from ingestion.splitters.text_splitter import TextSplitter
from ingestion.splitters.recursive_splitter import RecursiveTextSplitter
from ingestion.storage.storage import Storage
from ingestion.storage.weaviate import Weaviate
from ingestion.chunking.Chunker import Chunker
from constants import CHILD_CHUNKS_INDEX_NAME, PARENTS_CHUNK_INDEX_NAME
from retrieval.base_retrieval import Retrieval
from retrieval.sentence_window import SentenceWindowRetrieval
from retrieval.auto_merge import AutoMergeRetrieval
from util import get_dataclass_fields


class DocumentController:
    """
    Takes in text and splitter, embedding, storage objects
    Creates child and parent chunks, creates embeddings for child chunks, stores everything in storage
    """

    def __init__(self, splitter: TextSplitter = RecursiveTextSplitter(),
                 embedding: Embeddings = LocalEmbeddings(),
                 storage: Storage = Weaviate(),
                 retrieval: Retrieval = SentenceWindowRetrieval()):
        self.splitter = splitter
        self.embedding = embedding
        self.storage = storage
        self.retrieval = retrieval


    def process_text_and_store(self, text: str):
        logging.debug(f"Starting text chunking and store")
        chunker = Chunker()

        self.storage.delete_index(CHILD_CHUNKS_INDEX_NAME)
        self.storage.delete_index(PARENTS_CHUNK_INDEX_NAME)

        child_data_dict = get_dataclass_fields(ChildChunk)
        self.storage.create_new_index_if_not_exists(CHILD_CHUNKS_INDEX_NAME, child_data_dict)

        parent_data_dict = get_dataclass_fields(ParentChunk)
        self.storage.create_new_index_if_not_exists(PARENTS_CHUNK_INDEX_NAME, parent_data_dict)

        children_chunks = chunker.create_chunks_from_splits_children(chunker.split_text(text))
        children_chunks, parent_chunks = chunker.create_parent_chunks_using_child_chunks(children_chunks)

        children_chunks = self.embedding.get_embeddings_for_chunks(children_chunks)

        self.storage.add_data_to_index(CHILD_CHUNKS_INDEX_NAME, children_chunks)
        self.storage.add_data_to_index(PARENTS_CHUNK_INDEX_NAME, parent_chunks)


    def search_and_retrieve_result(self, query: str):
        logging.debug(f"Starting search and retrieve")

        vector = self.embedding.get_embedding(query)
        results = self.storage.hybrid_search(index_name=CHILD_CHUNKS_INDEX_NAME, query_vector=vector, query_str=query)
        final_context = self.retrieval.get_context(results)
        logging.debug(f"Final context retrieved : f{final_context}")

        self.storage.close_connection()

        return final_context

