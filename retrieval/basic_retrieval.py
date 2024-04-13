from typing import List

from ingestion.chunking.Chunk import ChildChunk
from retrieval.base_retrieval import BaseRetrieval


class BasicRetrieval(BaseRetrieval):

    def get_context(self, top_results: List[ChildChunk]):
        top_results = top_results[:5]
        context = []
        for chunk in top_results:

            context.append(chunk.text)

        return context
