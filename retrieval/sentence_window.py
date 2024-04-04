from retrieval.base_retrieval import BaseRetrieval


class SentenceWindowRetrieval(BaseRetrieval):

    def __init__(self, adjacent_neighbor_window_size: int = 1):
        """

        :param adjacent_neighbor_window_size: Final retrieved result is 2*adjacent_neighbor_window_size + 1
        """
        self.adjacent_neighbor_window_size = adjacent_neighbor_window_size

    def get_context(self, results, df):
        final_results = []

        for index in results:
            final_results.append(' '.join(df.iloc[index - self.adjacent_neighbor_window_size:index + self.adjacent_neighbor_window_size]['sentence']))

        return final_results
