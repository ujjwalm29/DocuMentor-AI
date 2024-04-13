from abc import ABC, abstractmethod


class BaseRetrieval(ABC):

    @abstractmethod
    def get_context(self, top_results):
        pass

