import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from retrieval.base_retrieval import BaseRetrieval


def get_similarity_query_dataframe(df: pd.DataFrame, query_embedding):
    df_embeddings = pd.DataFrame(df['embedding'].tolist())
    similarities = cosine_similarity(query_embedding, df_embeddings).flatten()

    top_indices = np.argsort(similarities)[::-1][:3]

    return top_indices


class LocalDataframe:

    def __init__(self, retrieval_strategy: BaseRetrieval = BaseRetrieval()):
        self.retrieval_strategy = retrieval_strategy

    def get_results(self, df: pd.DataFrame, query_embedding):
        top_indices = get_similarity_query_dataframe(df, query_embedding)

        return self.retrieval_strategy.get_context(top_indices, df)
