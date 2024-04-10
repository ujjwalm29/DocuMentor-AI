from embeddings.Embeddings import Embeddings
from embeddings.LocalHFEmbeddings import LocalEmbeddings
from ingestion.splitters.text_splitter import TextSplitter
from ingestion.splitters.recursive_splitter import RecursiveTextSplitter
from ingestion.storage.storage import Storage
from ingestion.storage.weaviateDB import WeaviateDB
from ingestion.chunking.Chunker import Chunker


class IngestionController:
    """
    Takes in text and splitter, embedding, storage objects
    Creates child and parent chunks, creates embeddings for child chunks, stores everything in storage
    """

    def __init__(self, splitter: TextSplitter = RecursiveTextSplitter(),
                 embedding: Embeddings = LocalEmbeddings(),
                 storage: Storage = WeaviateDB()):
        self.splitter = splitter
        self.embedding = embedding
        self.storage = storage


    def process_text_and_store(self, text: str):
        chunker = Chunker()

        children_chunks = chunker.create_chunks_from_splits_children(text)

        children_chunks, parent_chunks = chunker.create_parent_chunks_using_child_chunks(children_chunks)

        children_chunks = self.embedding.get_embeddings_for_chunks(children_chunks)
