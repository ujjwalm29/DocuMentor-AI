from abc import ABC, abstractmethod
from typing import List

from ingestion.chunking.Chunk import Chunk


class Retrieval(ABC):

    @abstractmethod
    def get_context(self, top_results: List[Chunk]) -> List[str]:
        pass

