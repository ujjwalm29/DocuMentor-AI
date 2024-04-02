import pandas as pd
from sentence_transformers import SentenceTransformer


class LocalEmbeddings:

    def __init__(
        self,
        embeddings_model: str = "mixedbread-ai/mxbai-embed-2d-large-v1"
    ):
        self.model = SentenceTransformer(embeddings_model)


    def process_row(self, row):
        cleaned_sentence = row['sentence'].replace("\n", " ")
        embedding = self.get_embedding(cleaned_sentence)
        return pd.Series([cleaned_sentence, embedding], index=['sentence', 'embedding'])


    def get_embeddings_for_dataframe(self, df: pd.DataFrame):

        # check if already processed in data/pkl/*

        df[['sentence', 'embedding']] = df.apply(self.process_row, axis=1)

        return df


    def get_embedding(self, input_str):
        return self.model.encode(input_str)
