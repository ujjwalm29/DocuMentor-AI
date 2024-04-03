from abc import ABC, abstractmethod


class BaseRetrieval:

    @abstractmethod
    def get_context(self, top_indices, df):
        return df.iloc[top_indices]['sentence']

