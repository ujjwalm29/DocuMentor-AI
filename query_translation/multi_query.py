import logging
from typing import List

from constants import CHILD_CHUNKS_INDEX_NAME
from embeddings.Embeddings import Embeddings
from generation.chat import Chat
from generation.groq_chat import ChatGroq
from ingestion.storage.storage import Storage
from query_translation.query_translator import QueryTranslator
from collections import defaultdict

logger = logging.getLogger(__name__)


class MultiQueryTranslator(QueryTranslator):
    """
    Create alternate queries from original query.
    Get context for all queries and apply Reciprocal Rank Fusion on results
    """

    def translate_query_and_generate_context(self, user_id, storage: Storage, query: str, embedding_gen: Embeddings,
                                             number_of_results: int = 20, query_properties: List[str] = "text"):

        # Generate alternate queries
        chat: Chat = ChatGroq()

        queries: List[str] = chat.get_multiple_queries(query)
        logger.debug(f"Alternate queries generated : {queries}")

        context = []
        chunk_scores = defaultdict(float)  # Default scores are 0.0
        chunks = {}  # To store the actual ChildChunk objects

        k = 60  # A typical value for RRF

        queries_with_embeddings = embedding_gen.get_embeddings_for_list(queries)

        # For each query, search, get context, apply RRF
        for gen_query in queries_with_embeddings:
            cur_context = storage.hybrid_search(user_id=user_id, index_name=CHILD_CHUNKS_INDEX_NAME,
                                                query_vector=gen_query['embedding'],
                                                query_str=gen_query['text'], number_of_results=number_of_results,
                                                query_properties=query_properties)

            # Apply Reciprocal Rank Fusion (RRF)
            for i, child_chunk in enumerate(cur_context):
                # The rank is i+1 because rank starts at 1
                chunk_scores[child_chunk.chunk_id] += 1 / (k + i + 1)
                chunks[child_chunk.chunk_id] = child_chunk  # Store the chunk by its ID

        # After processing all queries, we sort the chunks by their new aggregated scores
        sorted_chunk_ids = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)

        # Retrieve the actual chunk objects based on the sorted score
        final_results = [chunks[chunk_id] for chunk_id, _ in sorted_chunk_ids]

        context.extend(final_results)

        return context
