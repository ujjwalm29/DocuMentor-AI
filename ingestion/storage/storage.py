from abc import ABC, abstractmethod
from uuid import uuid4


class Storage(ABC):


    def __init__(self):
        super().__init__()

    @abstractmethod
    def create_new_index(self, index_name: str):
        pass


    @abstractmethod
    def get_index_size(self, index_name: str):
        pass


    @abstractmethod
    def delete_index(self, index_name: str):
        pass


    @abstractmethod
    def get_element_by_id(self, element_id: uuid4):
        pass


    @abstractmethod
    def vector_search(self, query_vector, number_of_results: int):
        pass


    @abstractmethod
    def hybrid_search(self, query_vector, query_str: str, number_of_results: int):
        pass