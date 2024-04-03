import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_similarity_query_dataframe(df: pd.DataFrame, query_embedding):
    df_embeddings = pd.DataFrame(df['embedding'].tolist())
    similarities = cosine_similarity(query_embedding, df_embeddings).flatten()

    top_indices = np.argsort(similarities)[::-1][:3]

    return top_indices


def sentence_window_retrieval(df: pd.DataFrame, query_embedding):
    top_indices = get_similarity_query_dataframe(df, query_embedding)

    final_results = []

    for index in top_indices:
        final_results.append(' '.join(df.iloc[index - 1:index + 1]['sentence']))

    return final_results
