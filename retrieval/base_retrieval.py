from abc import ABC, abstractmethod
from typing import List

from ingestion.chunking.Chunk import ChunkBase


class BaseRetrieval(ABC):

    @abstractmethod
    def get_context(self, top_results: List[ChunkBase]) -> List[str]:
        pass

