from collections import Counter
from typing import List
import logging

from ingestion.chunking.Chunk import ChildChunk
from ingestion.storage.storage import Storage
from ingestion.storage.weaviate import Weaviate
from retrieval.base_retrieval import Retrieval
from constants import PARENTS_CHUNK_INDEX_NAME

logger = logging.getLogger(__name__)


class AutoMergeRetrieval(Retrieval):

    def __init__(self, storage: Storage = Weaviate()):
        self.storage = storage

    def get_context(self, top_results: List[ChildChunk]):
        logger.debug(f"Auto Merge Retrieval get_context called with top_results ${top_results}")

        # Get all parent Ids
        parent_ids = [result.parent_id for result in top_results]

        parent_count_map = Counter(parent_ids)

        top_results = top_results[:10]

        used_parent = set()

        final_context = []

        for chunk in top_results:

            # check if parent is used
            if chunk.parent_id in used_parent:
                continue

            if parent_count_map.get(chunk.parent_id) > 2:
                parent_chunk = self.storage.get_element_by_chunk_id(index_name=PARENTS_CHUNK_INDEX_NAME,element_id=chunk.parent_id)

                final_context.append(parent_chunk.text)

                used_parent.add(chunk.parent_id)
            else:
                final_context.append(chunk.text)

            if len(final_context) >= 5:
                break

        self.storage.close_connection()  # To be removed when DB object is made singleton

        return final_context


