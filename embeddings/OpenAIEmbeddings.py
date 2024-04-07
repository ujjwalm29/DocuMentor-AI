import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import Embeddings

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class OpenAIEmbeddings(Embeddings):

    def __init__(
        self,
        file_path: str,
        embeddings_model: str = "text-embedding-3-small"
    ):
        self.file_path = file_path
        self.client = OpenAI(api_key=os.getenv("API_KEY"))
        self.model = embeddings_model


    def process_row(self, row):
        cleaned_sentence = row['sentence'].replace("\n", " ")
        embedding = self.get_embedding(cleaned_sentence)
        return pd.Series([cleaned_sentence, embedding], index=['sentence', 'embedding'])


    def get_embeddings_for_dataframe(self, df: pd.DataFrame):
        futures = []
        results = []
        with ThreadPoolExecutor() as executor:
            for _, row in df.iterrows():
                futures.append(executor.submit(self.process_row, row))

            for future in as_completed(futures):
                results.append(future.result())

        new_df = pd.DataFrame(results)
        df['sentence'], df['embedding'] = new_df['sentence'], new_df['embedding']

        if self.file_path != "":
            new_file_name = ''.join(self.file_path.split('/')[-1].split('.')[:-1]) + '.pkl'
            pkl_file_path = os.path.join(PROJECT_ROOT, 'data', 'pkl', new_file_name)
            df.to_pickle(pkl_file_path)

        return df


    def get_embedding(self, input_str):
        return self.client.embeddings.create(input=[input_str], model=self.model).data[0].embedding
