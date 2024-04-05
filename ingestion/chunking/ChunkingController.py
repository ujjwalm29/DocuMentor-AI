from ingestion.chunking.Chunk import ParentChunk, ChildChunk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embeddings.local import LocalEmbeddings
import uuid
import pandas as pd

from ingestion.pdf import PdfParser


class ChunkingController:

    def __init__(self):
        self.head_child_chunk = ChildChunk(uuid.uuid4().int, "", -1, -1, [], {}, -1)
        self.head_parent_chunk = ParentChunk(uuid.uuid4().int, "", -1, -1, [], {}, 0)
        self.parent_text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200)
        self.child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=300)
        self.child_parent_ratio = 4
        self.embedding_generator = LocalEmbeddings('not_needed')

    def create_chunks_from_splits_children(self, splits):
        uuid.uuid4()
        prev_chunk = self.head_child_chunk

        # Initialize an empty list to hold the chunk data
        chunks_data = []

        for split in splits:
            new_chunk_id = uuid.uuid4().int  # Generate a new unique ID
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

        # Convert the list of dictionaries to a DataFrame
        chunks_df = pd.DataFrame(chunks_data)

        # Set the 'id' column as the index of the DataFrame
        chunks_df.set_index('id', inplace=True)

        cur_chunk = self.head_child_chunk
        count = 0
        while cur_chunk.next_id != -1:
            # print(count)
            count += 1
            cur_chunk = chunks_df.loc[cur_chunk.next_id]

        print(count)
        print(len(splits))




    def split_text(self, text : str):
        # first split it into children and create a linked list of Chunks
        splits = self.child_text_splitter.split_text(text)

        self.create_chunks_from_splits_children(splits)

        # Then combine 4 at a time, create parent Chunk and assign parent_ids to child nodes
