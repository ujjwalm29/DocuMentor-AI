import os

import weaviate
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class WeaviateDB:


    def __init__(self, collection_name="pdf_store"):
        self.client = weaviate.connect_to_local(host=os.getenv("WEAVIATE_HOST"), port=int(os.getenv("WEAVIATE_PORT")))

        if not self.client.collections.exists(collection_name):
            self.collection = self.client.collections.create(collection_name)
        else:
            self.collection = self.client.collections.get(collection_name)

    def store(self, df: pd.DataFrame):
        with self.collection.batch.dynamic() as batch:
            for i, data_row in df.iterrows():
                batch.add_object(
                    properties={
                        "text": data_row['sentence']
                    },
                    vector=data_row['embedding']
                )


    def get_all_items(self):
        for item in self.collection.iterator():
            print(item.uuid, item.properties)


    def search_using_vector(self, query_vector):
        response = self.collection.query.near_vector(
            near_vector=query_vector,  # your query vector goes here
            limit=5
        )

        results = [entry.properties['text'] for entry in response.objects]

        return results
