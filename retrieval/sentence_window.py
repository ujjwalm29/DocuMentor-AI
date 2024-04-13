from ingestion.storage.weaviate import Weaviate
from ingestion.storage.storage import Storage
from retrieval.base_retrieval import BaseRetrieval
from constants import CHILD_CHUNKS_INDEX_NAME


class SentenceWindowRetrieval(BaseRetrieval):

    def __init__(self, storage: Storage = Weaviate(), adjacent_neighbor_window_size: int = 1):
        """

        :param adjacent_neighbor_window_size: Final retrieved result is 2*adjacent_neighbor_window_size + 1
        """
        self.adjacent_neighbor_window_size = adjacent_neighbor_window_size
        self.storage = storage

    def get_context(self, top_results):
        final_results = []

        for chunk in top_results:

            # Go back and forward in window size. Join everything
            context = []
            for i in range(0, self.adjacent_neighbor_window_size):
                if chunk.prev_id is None:
                    break
                prev_chunk = self.storage.get_element_by_chunk_id(CHILD_CHUNKS_INDEX_NAME, chunk.prev_id)
                context.append(prev_chunk.text)

            context.reverse()
            context.append(chunk.text)

            for i in range(0, self.adjacent_neighbor_window_size):
                if chunk.next_id is None:
                    break
                next_chunk = self.storage.get_element_by_chunk_id(CHILD_CHUNKS_INDEX_NAME, chunk.next_id)
                context.append(next_chunk.text)

            # Put in results
            final_results.append(''.join(context))

        return final_results
