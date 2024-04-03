from abc import ABC, abstractmethod


class TextSplitter:

    @abstractmethod
    def split_text(self, text: str):
        pass
