from abc import ABC, abstractmethod
import pandas as pd


class Embeddings(ABC):

    @abstractmethod
    def process_row(self, row):
        pass

    @abstractmethod
    def get_embeddings_for_dataframe(self, df: pd.DataFrame):
        pass

    @abstractmethod
    def get_embedding(self, input_str):
        pass
