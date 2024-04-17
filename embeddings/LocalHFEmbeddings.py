from typing import List

import logging
from sentence_transformers import SentenceTransformer
from embeddings.Embeddings import Embeddings
from ingestion.chunking.Chunk import ChunkBase

logger = logging.getLogger(__name__)


class LocalEmbeddings(Embeddings):

    def __init__(
        self,
        embeddings_model: str = "mixedbread-ai/mxbai-embed-2d-large-v1",
    ):
        logger.debug(f"Local HF Embeddings initialized. Model name = {embeddings_model}")
        self.model = SentenceTransformer(embeddings_model)


    def get_embeddings_for_chunks(self, chunks: List[ChunkBase]):
        texts = [obj.text for obj in chunks]

        embeddings = self.model.encode(texts, batch_size=32)

        for obj, embedding in zip(chunks, embeddings):
            obj.embeddings = embedding

        return chunks


    def get_embedding(self, input_str):
        return self.model.encode(input_str).tolist()
