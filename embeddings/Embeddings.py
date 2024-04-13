from abc import ABC, abstractmethod
from typing import List

import pandas as pd
from ingestion.chunking.Chunk import Chunk


class Embeddings(ABC):

    @abstractmethod
    def get_embeddings_for_chunks(self, chunks: List[Chunk]):
        pass

    @abstractmethod
    def get_embedding(self, input_str):
        pass
