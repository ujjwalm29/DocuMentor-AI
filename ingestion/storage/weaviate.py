from uuid import uuid4

from ingestion.storage.storage import Storage


class Weaviate(Storage):

    def __init__(self):
        super().__init__()
        pass

    def create_new_index(self, index_name: str):
        pass

    def get_index_size(self, index_name: str):
        pass

    def delete_index(self, index_name: str):
        pass

    def get_element_by_id(self, element_id: uuid4):
        pass

    def vector_search(self, query_vector, number_of_results: int):
        response = self.collection.query.near_vector(
            near_vector=query_vector,
            limit=5
        )

        results = [entry.properties['text'] for entry in response.objects]

        return results

    def hybrid_search(self, query_vector, query_str: str, number_of_results: int):
        response = self.collection.query.hybrid(
            query=query_str,
            query_properties=[query_properties],
            vector=query_vector,
            limit=5
        )

        results = [entry.properties['text'] for entry in response.objects]

        return results