from retrieval.base_retrieval import BaseRetrieval


class SentenceWindowRetrieval(BaseRetrieval):

    def get_context(self, results, df):
        final_results = []

        for index in results:
            final_results.append(' '.join(df.iloc[index - 1:index + 1]['sentence']))

        return final_results
