from typing import List

from ingestion.chunking.Chunk import ParentChunk, ChildChunk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embeddings.local import LocalEmbeddings
import pandas as pd
import itertools
import random


def create_custom_id_generator(min_id=-2147483648, max_id=2147483647):
    """
    Creates a generator that yields unique IDs within the specified range.
    The generator combines a sequential counter and a random offset to produce IDs.

    :param min_id: Minimum possible ID (inclusive).
    :param max_id: Maximum possible ID (inclusive).
    :return: A generator yielding unique IDs within the specified range.
    """
    counter = itertools.count()
    range_size = max_id - min_id + 1

    # Function to generate an ID
    def generate_id():
        # Get the next count and apply a random offset within the range
        base_id = next(counter)
        random_offset = random.randint(0, range_size - 1)
        # Combine the counter and random offset, ensuring the result fits within the range
        return ((base_id + random_offset) % range_size) + min_id

    # Return the generate_id function as a generator
    return generate_id




class ChunkingController:

    def __init__(self):
        self.id_generator = create_custom_id_generator()

        self.head_child_chunk = ChildChunk(self.id_generator(), "", -1, -1, [], {}, -1)
        self.tail_child_chunk = ChildChunk(self.id_generator(), "", -1, -1, [], {}, -1)
        self.head_child_chunk.next_id = self.tail_child_chunk.id
        self.tail_child_chunk.next_id = self.head_child_chunk.id

        self.head_parent_chunk = ParentChunk(self.id_generator(), "", -1, -1, [], {}, 0)
        self.tail_parent_chunk = ParentChunk(self.id_generator(), "", -1, -1, [], {}, 0)
        self.head_parent_chunk.next_id = self.tail_parent_chunk.id
        self.tail_parent_chunk.next_id = self.head_parent_chunk.id

        self.parent_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200)
        self.child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=300)
        self.child_parent_ratio = 4
        self.embedding_generator = LocalEmbeddings('not_needed')

    def create_chunks_from_splits_children(self, splits):
        prev_chunk = self.head_child_chunk

        # Initialize an empty list to hold the chunk data
        chunks_data = []

        for split in splits:
            new_chunk_id = self.id_generator()  # Generate a new unique ID
            prev_chunk.next_id = new_chunk_id
            new_chunk = ChildChunk(
                id=new_chunk_id,
                text=split,
                prev_id=prev_chunk.id,
                next_id=-1,  # Will be updated later or remains -1 if it's the last chunk
                embeddings=[],
                metadata={},
                parent_id=-1
            )
            chunks_data.append(new_chunk)  # Add the new chunk to the list
            prev_chunk = new_chunk  # Update prev_chunk for the next iteration

        prev_chunk.next_id = self.tail_child_chunk.id
        self.tail_child_chunk.prev_id = prev_chunk.id
        chunks_data.append(self.tail_child_chunk)  # might remove later

        # Convert the list of dictionaries to a DataFrame
        chunks_df = pd.DataFrame(chunks_data)

        # Set the 'id' column as the index of the DataFrame
        chunks_df.set_index('id', inplace=True)

        cur_chunk = chunks_df.loc[self.head_child_chunk.next_id]
        count = 0
        while cur_chunk.name != self.tail_child_chunk.id:
            count += 1
            cur_chunk = chunks_df.loc[cur_chunk.next_id]

        print(count)
        print(len(splits))
        assert count == len(splits)

        return chunks_df

    def split_text(self, text: str):
        # first split it into children and create a linked list of Chunks
        splits = self.child_text_splitter.split_text(text)

        children_chunks_df = self.create_chunks_from_splits_children(splits)

        # Then combine 4 at a time, create parent Chunk and assign parent_ids to child nodes
        self.create_parent_chunks_using_child_chunks(children_chunks_df)

    def create_parent_chunks_using_child_chunks(self, children_chunks_df):

        children_chunks_df['parent_id'] = pd.to_numeric(children_chunks_df['parent_id'], errors='coerce')

        prev_par_chunk = self.head_parent_chunk
        cur_child_chunk = children_chunks_df.loc[self.head_child_chunk.next_id]
        parent_chunks = []

        while cur_child_chunk.name != self.tail_child_chunk.id:
            # get 4 or less chunks
            child_chunks, cur_child_chunk = self.get_4_or_less_chunks(cur_child_chunk, children_chunks_df)

            # Create parent chunk ID
            parent_chunk_ID = self.id_generator()

            text = ""
            # Combine all texts and set parent_id
            for child_chunk in child_chunks:
                text = text + child_chunk.text
                children_chunks_df.loc[child_chunk.name, 'parent_id'] = parent_chunk_ID

            # Create parent_chunk
            new_parent_chunk = ParentChunk(
                id=parent_chunk_ID,
                text=text,
                embeddings=[],
                metadata={},
                prev_id=prev_par_chunk.id,
                next_id=-1,
                number_of_children=len(child_chunks)
            )

            prev_par_chunk.next_id = parent_chunk_ID
            prev_par_chunk = new_parent_chunk
            parent_chunks.append(new_parent_chunk)

        prev_par_chunk.next_id = self.tail_parent_chunk.id
        self.tail_parent_chunk.prev_id = prev_par_chunk.id
        parent_chunks.append(self.tail_parent_chunk)  # might remove later

        # Convert the list of dictionaries to a DataFrame
        chunks_df = pd.DataFrame(parent_chunks)

        # Set the 'id' column as the index of the DataFrame
        chunks_df.set_index('id', inplace=True)

        cur_chunk = chunks_df.loc[self.head_parent_chunk.next_id]
        count = 0
        while cur_chunk.name != self.tail_parent_chunk.id:
            count += 1
            cur_chunk = chunks_df.loc[cur_chunk.next_id]

        print(count)
        print(len(children_chunks_df) / 4)

        return chunks_df

    def get_4_or_less_chunks(self, cur_child_chunk, chunks_df):
        sub_child_chunks = []
        count = 0

        while count < 4 and cur_child_chunk.name != self.tail_child_chunk.id:
            count += 1
            sub_child_chunks.append(cur_child_chunk)
            cur_child_chunk = chunks_df.loc[cur_child_chunk.next_id]

        return sub_child_chunks, cur_child_chunk


