from abc import ABC, abstractmethod
from typing import List

from embeddings.Embeddings import Embeddings
from ingestion.storage.storage import Storage


class QueryTranslator(ABC):

    @abstractmethod
    def translate_query_and_generate_context(self, user_id, storage: Storage, query: str, embedding_gen: Embeddings, number_of_results: int, query_properties: List[str]):
        pass