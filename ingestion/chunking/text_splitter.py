from abc import ABC, abstractmethod


class TextSplitter(ABC):

    @abstractmethod
    def split_text(self, text: str):
        pass
