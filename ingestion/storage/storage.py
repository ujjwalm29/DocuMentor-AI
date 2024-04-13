from abc import ABC, abstractmethod
from typing import List, Dict
from uuid import uuid4


class Storage(ABC):


    def __init__(self):
        super().__init__()

    @abstractmethod
    def create_new_index_if_not_exists(self, index_name: str, properties: Dict):
        pass

    @abstractmethod
    def add_data_to_index(self, index_name: str, data: List):
        pass


    @abstractmethod
    def get_index_size(self, index_name: str):
        pass


    @abstractmethod
    def delete_index(self, index_name: str):
        pass


    @abstractmethod
    def get_element_by_chunk_id(self, index_name: str, element_id: uuid4):
        pass


    @abstractmethod
    def vector_search(self, index_name: str, query_vector, number_of_results: int):
        pass


    @abstractmethod
    def hybrid_search(self, index_name: str, query_vector, query_str: str, number_of_results: int, query_properties: List[str]):
        pass


    @abstractmethod
    def close_connection(self):
        pass
