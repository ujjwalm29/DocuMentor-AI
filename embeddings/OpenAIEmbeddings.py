from typing import List

from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from embeddings.Embeddings import Embeddings
from ingestion.chunking.Chunk import ChunkBase

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class OpenAIEmbeddings(Embeddings):

    def __init__(
        self,
        embeddings_model: str = "text-embedding-3-small"
    ):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = embeddings_model

    def get_embeddings_for_chunks(self, chunks: List[ChunkBase]):
        texts = [obj.text for obj in chunks]

        with ThreadPoolExecutor() as executor:
            # Submit all tasks to the executor
            future_to_text = {executor.submit(self.get_embedding, text): text for text in texts}

            # Collect the results as they are completed
            results = []
            for future in as_completed(future_to_text):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    print(f'Generated an exception: {exc}')

        for obj, embedding in zip(chunks, results):
            obj.embeddings = embedding

        return chunks

    def get_embedding(self, input_str):
        return self.client.embeddings.create(input=[input_str], model=self.model).data[0].embedding

    def get_embeddings_for_list(self, list_of_str: List[str]):
        raise NotImplementedError()
        pass
