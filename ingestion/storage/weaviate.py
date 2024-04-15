import datetime
import os
from datetime import time
from typing import List, Dict
from uuid import uuid4, UUID
import logging

import weaviate
from ingestion.storage.storage import Storage
from dotenv import load_dotenv
from weaviate.classes.config import Property, DataType
from weaviate.classes.query import Filter
from ingestion.chunking.Chunk import ChildChunk, ParentChunk
from constants import CHILD_CHUNKS_INDEX_NAME, PARENTS_CHUNK_INDEX_NAME, DOCUMENT_INDEX_NAME
from dataclasses import asdict, is_dataclass

logger = logging.getLogger(__name__)

load_dotenv()

property_name_map = {
    str: DataType.TEXT,
    UUID: DataType.UUID,
    UUID | None: DataType.UUID,
    object: DataType.TEXT,
    int: DataType.INT,
    time: DataType.DATE
}


class Weaviate(Storage):

    def __init__(self):
        super().__init__()
        logger.info("Initializing connection to weaviate database")
        self.client = weaviate.connect_to_local(host=os.getenv("WEAVIATE_HOST"), port=int(os.getenv("WEAVIATE_PORT")))


    def create_new_index(self, index_name: str, index_properties: Dict):
        logger.info(f"Create new index with name {index_name}")
        logger.debug(f"Index properties {index_properties}")
        properties = list()
        for field_name, field_type in index_properties.items():
            if field_name == "embeddings":
                continue

            new_prop = Property(name=field_name, data_type=property_name_map[field_type])
            if field_type == str:
                new_prop.indexSearchable = True

            properties.append(new_prop)

        self.client.collections.create(index_name, properties=properties)
        logger.info(f"{index_name} created!")


    def create_new_index_if_not_exists(self, index_name: str, index_properties: Dict):
        if not self.client.collections.exists(index_name):
            self.create_new_index(index_name, index_properties)
        else:
            logger.info(f"{index_name} already exists")

    def get_index_size(self, index_name: str = CHILD_CHUNKS_INDEX_NAME):
        data = self.client.collections.get(index_name)

        count = sum(1 for _ in data.iterator())

        return count

    def add_data_to_index(self, index_name: str, data: List):
        logger.info(f"Adding data to index {index_name}")
        logger.debug(f"Data {data}")
        collection = self.client.collections.get(index_name.lower())

        with collection.batch.dynamic() as batch:
            for item in data:
                if is_dataclass(item):
                    props = asdict(item)
                    obj_vector = asdict(item).pop('embeddings')

                    property_value = {
                        "text": props['text'],
                        "prev_id": props['prev_id'],
                        "next_id": props['next_id'],
                        "user_id": props['user_id'],
                        "document_id": props['document_id']

                    }

                    if index_name.lower() == CHILD_CHUNKS_INDEX_NAME:
                        property_value['parent_id'] = props['parent_id']
                    else:
                        property_value['number_of_children'] = props['number_of_children']

                    batch.add_object(properties=property_value, vector=obj_vector, uuid=props['chunk_id'])




    def delete_index(self, index_name: str):
        logger.warning(f"delete_index called for Index {index_name}")
        if self.client.collections.exists(index_name):
            self.client.collections.delete(index_name)

    def get_element_by_chunk_id(self, index_name: str, element_id: UUID) -> ChildChunk:
        logger.debug(f"get_element_by_chunk_id called for ID {element_id} Index {index_name}")
        element = self.client.collections.get(index_name).query.fetch_object_by_id(element_id)

        return self.create_chunk_from_weaviate_objects(index_name, element)

    def vector_search(self, user_id: UUID, index_name: str, query_vector, number_of_results: int = 20):
        logger.debug(f"vector search Index {index_name}")
        response = self.client.collections.get(index_name).query.near_vector(
            filters=Filter.by_property("user_id").equal(user_id),
            near_vector=query_vector.tolist(),
            limit=number_of_results
        )

        # results = [entry.properties['text'] for entry in response.objects]

        response_chunks = []

        for obj in response.objects:
            response_chunks.append(self.create_chunk_from_weaviate_objects(index_name, obj))

        return response_chunks

    def hybrid_search(self, user_id: UUID, index_name: str, query_vector, query_str: str, number_of_results: int = 20, query_properties: List[str]="text"):
        logger.debug(f"hybrid search Index {index_name}, Query {query_str}, Query Properties {query_properties}")
        response = self.client.collections.get(index_name).query.hybrid(
            filters=Filter.by_property("user_id").equal(user_id),
            query=query_str,
            query_properties=[query_properties],
            vector=query_vector.tolist(),
            limit=number_of_results
        )

        response_chunks = []

        for obj in response.objects:
            response_chunks.append(self.create_chunk_from_weaviate_objects(index_name, obj))

        return response_chunks


    def add_doc_for_user(self, name: str, document_id: UUID, user_id: UUID):
        self.client.collections.get(DOCUMENT_INDEX_NAME).data.insert({
            "name": name,
            "document_id": document_id,
            "user_id": user_id,
            "added_at": datetime.datetime.now()
        })


    def delete_doc_from_db(self, document_id: UUID, user_id: UUID):
        # TODO : Check if document_id belongs to user
        self.client.collections.get(DOCUMENT_INDEX_NAME).data.delete_many(
            where=Filter.by_property("document_id").contains_any([document_id])
        )


    def delete_chunks_for_doc_id(self, document_id: UUID, user_id: UUID):
        # TODO : Implement this
        pass


    def close_connection(self):
        logger.info("Closing weaviate client connection")
        self.client.close()


    def create_chunk_from_weaviate_objects(self, index_name, obj):
        if index_name == CHILD_CHUNKS_INDEX_NAME:
            return ChildChunk(
                chunk_id=obj.uuid,
                parent_id=obj.properties['parent_id'],
                next_id=obj.properties['next_id'],
                prev_id=obj.properties['prev_id'],
                text=obj.properties['text'],
                embeddings=[],
                metadata={},
                user_id=obj.properties['user_id'],
                document_id=obj.properties['document_id']
            )
        else:
            return ParentChunk(
                chunk_id=obj.uuid,
                number_of_children=obj.properties['number_of_children'],
                next_id=obj.properties['next_id'],
                prev_id=obj.properties['prev_id'],
                text=obj.properties['text'],
                embeddings=[],
                metadata={},
                user_id=obj.properties['user_id'],
                document_id=obj.properties['document_id']
            )
