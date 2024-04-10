from abc import ABC, abstractmethod
from typing import List

import pandas as pd
from ingestion.chunking.Chunk import ChunkBase


class Embeddings(ABC):

    @abstractmethod
    def get_embeddings_for_chunks(self, chunks: List[ChunkBase]):
        pass

    @abstractmethod
    def get_embedding(self, input_str):
        pass
