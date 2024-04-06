from typing import List

from ingestion.chunking.Chunk import ParentChunk, ChildChunk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embeddings.local import LocalEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
import itertools
import random
import os
from collections import Counter


def create_custom_id_generator(min_id=0, max_id=2147483647):
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
        self.embedding_generator = LocalEmbeddings('not_needed')

        self.head_child_chunk = ChildChunk(self.id_generator(), "", -1, -1, [], {}, -1)
        self.tail_child_chunk = ChildChunk(self.id_generator(), "", -1, -1, self.embedding_generator.get_embedding("random"), {}, -1)
        self.head_child_chunk.next_id = self.tail_child_chunk.id
        self.tail_child_chunk.next_id = self.head_child_chunk.id

        self.head_parent_chunk = ParentChunk(self.id_generator(), "", -1, -1, [], {}, 0)
        self.tail_parent_chunk = ParentChunk(self.id_generator(), "", -1, -1, [], {}, 0)
        self.head_parent_chunk.next_id = self.tail_parent_chunk.id
        self.tail_parent_chunk.next_id = self.head_parent_chunk.id

        self.child_parent_ratio = 4
        self.child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
        self.parent_text_splitter = RecursiveCharacterTextSplitter(chunk_size=200*self.child_parent_ratio, chunk_overlap=0)


    def create_chunks_from_splits_children(self, splits):
        prev_chunk = self.head_child_chunk

        # Initialize an empty list to hold the chunk data
        chunks_data = []

        for split in splits:
            new_chunk_id = self.id_generator()  # Generate a new unique ID
            prev_chunk.next_id = new_chunk_id

            if split == "":
                continue

            new_chunk = ChildChunk(
                id=new_chunk_id,
                text=split,
                prev_id=prev_chunk.id,
                next_id=-1,  # Will be updated later or remains -1 if it's the last chunk
                embeddings=self.embedding_generator.get_embedding(split),
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

        for split, i in enumerate(splits):
            print(f"{i} -------------- \n{split}")


        if os.path.exists('./children.pkl'):
            children_chunks_df = pd.read_pickle('./children.pkl')
        else:
            children_chunks_df = self.create_chunks_from_splits_children(splits)



        # Then combine 4 at a time, create parent Chunk and assign parent_ids to child nodes
        if os.path.exists('./parent.pkl'):
            parent_chunks_df = pd.read_pickle('./parent.pkl')
        else:
            parent_chunks_df = self.create_parent_chunks_using_child_chunks(children_chunks_df)
            parent_chunks_df.to_pickle('./parent.pkl')
            children_chunks_df.to_pickle('./children.pkl')

        # print(children_chunks_df.iloc[0])
        # print(parent_chunks_df.iloc[0])

        #self.auto_merging_retrieval("What are the parameters N,R,W in Dynamo?", children_chunks_df, parent_chunks_df)
        self.sentence_window_retrieval("What are the parameters N,R,W in Dynamo?", children_chunks_df, 1)

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


    def auto_merging_retrieval(self, query: str, children_chunks_df: pd.DataFrame, parent_chunks_df: pd.DataFrame):
        df_embeddings = pd.DataFrame(children_chunks_df['embeddings'].tolist())
        similarities = cosine_similarity(self.embedding_generator.get_embedding([query]), df_embeddings).flatten()

        top_indices = np.argsort(similarities)[::-1][:30]

        # Now, it's time to merge
        # Get a map Of parent IDs, count_of_children
        # Get the integer position of the 'parent_id' column
        parent_id_col_pos = children_chunks_df.columns.get_loc('parent_id')

        # Use .iloc with top_indices and the column position to access 'parent_id' values
        parent_ids = children_chunks_df.iloc[top_indices, parent_id_col_pos]

        # Use Counter to count the frequency of each parent_id
        parent_count_map = Counter(parent_ids)

        print(parent_count_map)

        # In final results, sort by top 5. If parent has count>2, don't add child, add parent
        # Store info so parent or any child of parent don't get added to the top 5.
        top_indices = top_indices[:10]

        used_parent = set()

        final_context = []

        for index in top_indices:
            parent_id = children_chunks_df.iloc[index]['parent_id']

            if parent_id in used_parent:
                continue

            if parent_count_map[parent_id] > 2:
                print("Putting parent..")
                final_context.append(parent_chunks_df.loc[parent_id, 'text'])
                used_parent.add(parent_id)
            else:
                final_context.append(children_chunks_df.iloc[index]['text'])

            if len(final_context) >= 5:
                break


        return final_context


    def sentence_window_retrieval(self, query: str, children_chunks_df: pd.DataFrame, window_size: int):
        df_embeddings = pd.DataFrame(children_chunks_df['embeddings'].tolist())
        similarities = cosine_similarity(self.embedding_generator.get_embedding([query]), df_embeddings).flatten()

        top_indices = np.argsort(similarities)[::-1][:5]

        print(top_indices)

        final_context = []

        # parse from node to back, node to front using IDs. Make sure to keep track of order.
        for index in top_indices:
            cur_node = children_chunks_df.iloc[index]

            prev_node = children_chunks_df.loc[cur_node['prev_id']]
            context_before = []
            while len(context_before) < window_size and prev_node.name != self.head_child_chunk.id:
                context_before.append(prev_node['text'])
                prev_node = children_chunks_df.loc[prev_node['prev_id']]

            context_before.reverse()

            next_node = children_chunks_df.loc[cur_node['next_id']]
            context_after = []
            while len(context_after) < window_size and next_node.name != self.tail_child_chunk.id:
                context_after.append(next_node['text'])
                next_node = children_chunks_df.loc[next_node['next_id']]

            print("context before")
            for context in context_before:
                print(context)

            print("From DF")

            print(children_chunks_df.iloc[index-1]['text'])

            print("---")

            print()
            print("cur")
            print(cur_node.text)
            print("From DF")
            print(children_chunks_df.iloc[index]['text'])
            print("---")

            print()
            print("context after")
            for context in context_after:
                print(context)

            print()
            print("From DF")
            print(children_chunks_df.iloc[index+1]['text'])
            print("---")

            concat_context = ''.join(context_before) + cur_node.text + ''.join(context_after)

            final_context.append(concat_context)

        for context in final_context:
            print()
            print(context)

        return final_context

        # send results


