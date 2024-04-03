import pandas as pd
from sentence_transformers import SentenceTransformer
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class LocalEmbeddings:

    def __init__(
        self,
        file_path: str,
        embeddings_model: str = "mixedbread-ai/mxbai-embed-2d-large-v1",
    ):
        if file_path == "":
            raise AssertionError('file_path cannot be None')
        self.file_name = file_path
        self.model = SentenceTransformer(embeddings_model)


    def process_row(self, row):
        cleaned_sentence = row['sentence'].replace("\n", " ")
        embedding = self.get_embedding(cleaned_sentence)
        return pd.Series([cleaned_sentence, embedding], index=['sentence', 'embedding'])


    def get_embeddings_for_dataframe(self, df: pd.DataFrame):
        new_file_name = ''.join(self.file_name.split('/')[-1].split('.')[:-1]) + '.pkl'
        pkl_file_path = os.path.join(PROJECT_ROOT, 'data', 'pkl', new_file_name)

        df[['sentence', 'embedding']] = df.apply(self.process_row, axis=1)
        df.to_pickle(pkl_file_path)

        return df


    def get_embedding(self, input_str):
        return self.model.encode(input_str)
