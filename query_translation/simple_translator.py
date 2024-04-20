from typing import List

from constants import CHILD_CHUNKS_INDEX_NAME
from embeddings.Embeddings import Embeddings
from ingestion.storage.storage import Storage
from query_translation.query_translator import QueryTranslator


class SimpleTranslator(QueryTranslator):

    def translate_query_and_generate_context(self, user_id, storage: Storage, query: str, embedding_gen: Embeddings,
                                             number_of_results: int = 20, query_properties: List[str] = "text"):
        vector = embedding_gen.get_embedding(query)
        context = storage.hybrid_search(user_id=user_id, index_name=CHILD_CHUNKS_INDEX_NAME, query_vector=vector,
                                        query_str=query, number_of_results=number_of_results, query_properties=query_properties)

        return context
