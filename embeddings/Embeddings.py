from abc import ABC, abstractmethod
from typing import List

from ingestion.chunking.Chunk import ChunkBase


class Embeddings(ABC):

    @abstractmethod
    def get_embeddings_for_chunks(self, chunks: List[ChunkBase]):
        pass

    @abstractmethod
    def get_embedding(self, input_str):
        pass

    @abstractmethod
    def get_embeddings_for_list(self, list_of_str: List[str]):
        pass
