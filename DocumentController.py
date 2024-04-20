import logging
import uuid
from uuid import UUID

from embeddings.Embeddings import Embeddings
from embeddings.LocalHFEmbeddings import LocalEmbeddings
from ingestion.chunking.Chunk import ChildChunk, ParentChunk, Document
from ingestion.pdf import PdfParser
from ingestion.splitters.text_splitter import TextSplitter
from ingestion.splitters.recursive_splitter import RecursiveTextSplitter
from ingestion.storage.storage import Storage
from ingestion.storage.weaviate import Weaviate
from ingestion.chunking.Chunker import Chunker
from constants import CHILD_CHUNKS_INDEX_NAME, PARENTS_CHUNK_INDEX_NAME, DOCUMENT_INDEX_NAME
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
                 retrieval: Retrieval = SentenceWindowRetrieval(adjacent_neighbor_window_size=1),
                 pdf_parser: PdfParser = PdfParser(parsing_instructs="""
You are parsing educational information. DO NOT include the references section or any metadata or acknowledgments in the output.
Convert tables into a list of facts. Do not include "Research Paper" or any other term repeatedly in the final md file.
""")):
        logging.info("Init Doc Controller")
        self.splitter = splitter
        self.embedding = embedding
        self.storage = storage
        self.retrieval = retrieval
        self.pdf_parser = pdf_parser


    async def process_text_and_store(self, file_path: str, user_id: UUID = UUID("a84080ee-b3d3-4269-bb8a-ff3887a90edb")):
        logging.debug(f"Starting document parsing, text chunking and store")

        text = await self.pdf_parser.get_text_from_pdf(file_path)

        chunker = Chunker(self.embedding)

        print(user_id)
        doc_id = uuid.uuid4()

        self.create_indexes_if_not_exist()

        self.add_doc_user_to_db(file_path.split('/')[-1], doc_id, user_id)

        children_chunks = chunker.create_chunks_from_splits_children(doc_id, user_id, chunker.split_text(text))
        children_chunks, parent_chunks = chunker.create_parent_chunks_using_child_chunks(doc_id, user_id, children_chunks)

        children_chunks = self.embedding.get_embeddings_for_chunks(children_chunks)

        self.storage.add_data_to_index(CHILD_CHUNKS_INDEX_NAME, children_chunks)
        self.storage.add_data_to_index(PARENTS_CHUNK_INDEX_NAME, parent_chunks)


    def search_and_retrieve_result(self, query: str, user_id: UUID = UUID("a84080ee-b3d3-4269-bb8a-ff3887a90edb")):
        logging.debug(f"Starting search and retrieve")

        vector = self.embedding.get_embedding(query)
        results = self.storage.hybrid_search(user_id=user_id, index_name=CHILD_CHUNKS_INDEX_NAME, query_vector=vector, query_str=query)
        final_context = self.retrieval.get_context(results)
        logging.debug(f"Final context retrieved : f{final_context}")

        return final_context


    def delete_indexes(self):
        self.storage.delete_index(CHILD_CHUNKS_INDEX_NAME)
        self.storage.delete_index(PARENTS_CHUNK_INDEX_NAME)
        self.storage.delete_index(DOCUMENT_INDEX_NAME)


    def create_indexes_if_not_exist(self):
        child_data_dict = get_dataclass_fields(ChildChunk)
        self.storage.create_new_index_if_not_exists(CHILD_CHUNKS_INDEX_NAME, child_data_dict)

        parent_data_dict = get_dataclass_fields(ParentChunk)
        self.storage.create_new_index_if_not_exists(PARENTS_CHUNK_INDEX_NAME, parent_data_dict)

        document_data_dict = get_dataclass_fields(Document)
        self.storage.create_new_index_if_not_exists(DOCUMENT_INDEX_NAME, document_data_dict)


    def add_doc_user_to_db(self, name, doc_id, user_id):
        self.storage.add_doc_for_user(name, doc_id, user_id)
