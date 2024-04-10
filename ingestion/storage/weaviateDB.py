import os

import weaviate
import pandas as pd
from dotenv import load_dotenv
from weaviate.classes.config import Property, DataType

load_dotenv()


class WeaviateDB:


    def __init__(self, text_properties=None, collection_name="pdf_store"):
        if text_properties is None:
            text_properties = ["text"]
        self.client = weaviate.connect_to_local(host=os.getenv("WEAVIATE_HOST"), port=int(os.getenv("WEAVIATE_PORT")))
        self.collection_name = collection_name

        if not self.client.collections.exists(collection_name):
            properties = list()
            for prop in text_properties:
                properties.append(
                    Property(name=prop, data_type=DataType.TEXT, index_searchable=True)
                )

            self.collection = self.client.collections.create(collection_name, properties=properties)
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
            near_vector=query_vector,
            limit=5
        )

        results = [entry.properties['text'] for entry in response.objects]

        return results

    def search_hybrid(self, query, query_vector, query_properties):
        response = self.collection.query.hybrid(
            query=query,
            query_properties=[query_properties],
            vector=query_vector,
            limit=5
        )

        results = [entry.properties['text'] for entry in response.objects]

        return results

    def delete_collection(self):
        self.client.collections.delete(self.collection_name)


    def close_connection(self):
        self.client.close()
