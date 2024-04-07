from abc import ABC, abstractmethod


class BaseRetrieval(ABC):

    @abstractmethod
    def get_context(self, top_indices, df):
        final_results = []

        for index in top_indices:
            final_results.append(df.iloc[index]['sentence'])

        return final_results

