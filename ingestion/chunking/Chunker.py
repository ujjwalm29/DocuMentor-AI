from typing import List

from ingestion.chunking.Chunk import ParentChunk, ChildChunk
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embeddings.LocalHFEmbeddings import LocalEmbeddings
import math
import uuid


class Chunker:

    def __init__(self):
        self.embedding_generator = LocalEmbeddings('not_needed')

        self.child_parent_ratio = 4
        self.child_text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)


    def create_chunks_from_splits_children(self, splits):
        prev_chunk = None

        # Initialize an empty list to hold the chunk data
        chunks_data = []

        for split in splits:
            new_chunk_id = uuid.uuid4()  # Generate a new unique ID

            if prev_chunk is not None:
                prev_chunk.next_id = new_chunk_id

            if split == "":
                continue

            new_chunk = ChildChunk(
                id=new_chunk_id,
                text=split,
                prev_id=None if prev_chunk is None else prev_chunk.id,
                next_id=None,  # Will be updated later or remains -1 if it's the last chunk
                embeddings=[], # self.embedding_generator.get_embedding(split),
                metadata={},
                parent_id=None
            )
            chunks_data.append(new_chunk)  # Add the new chunk to the list
            prev_chunk = new_chunk  # Update prev_chunk for the next iteration

        assert len(chunks_data) == len(splits)

        return chunks_data

    def create_parent_chunks_using_child_chunks(self, children_chunks_list: List[ChildChunk]):

        prev_par_chunk = None
        parent_chunks = []

        chunk_size = 4
        for i in range(0, len(children_chunks_list), chunk_size):
            # Get the next chunk of up to 4 elements
            child_chunks = children_chunks_list[i:i + chunk_size]

            parent_chunk_ID = uuid.uuid4()
            text = ""
            for chunk in child_chunks:
                chunk.parent_id = parent_chunk_ID
                text += chunk.text

            new_parent_chunk = ParentChunk(
                id=parent_chunk_ID,
                text=text,
                embeddings=[],
                metadata={},
                prev_id=None if prev_par_chunk is None else prev_par_chunk.id,
                next_id=None,
                number_of_children=len(child_chunks)
            )

            # If you need to replace the original list elements with modified ones:
            children_chunks_list[i:i + chunk_size] = child_chunks

            parent_chunks.append(new_parent_chunk)

            prev_par_chunk = new_parent_chunk


        assert math.ceil(len(children_chunks_list) / self.child_parent_ratio) == len(parent_chunks)

        return children_chunks_list, parent_chunks


    # def auto_merging_retrieval(self, query: str, children_chunks_df: pd.DataFrame, parent_chunks_df: pd.DataFrame):
    #     df_embeddings = pd.DataFrame(children_chunks_df['embeddings'].tolist())
    #     similarities = cosine_similarity(self.embedding_generator.get_embedding([query]), df_embeddings).flatten()
    #
    #     top_indices = np.argsort(similarities)[::-1][:30]
    #
    #     # Now, it's time to merge
    #     # Get a map Of parent IDs, count_of_children
    #     # Get the integer position of the 'parent_id' column
    #     parent_id_col_pos = children_chunks_df.columns.get_loc('parent_id')
    #
    #     # Use .iloc with top_indices and the column position to access 'parent_id' values
    #     parent_ids = children_chunks_df.iloc[top_indices, parent_id_col_pos]
    #
    #     # Use Counter to count the frequency of each parent_id
    #     parent_count_map = Counter(parent_ids)
    #
    #     print(parent_count_map)
    #
    #     # In final results, sort by top 5. If parent has count>2, don't add child, add parent
    #     # Store info so parent or any child of parent don't get added to the top 5.
    #     top_indices = top_indices[:10]
    #
    #     used_parent = set()
    #
    #     final_context = []
    #
    #     for index in top_indices:
    #         parent_id = children_chunks_df.iloc[index]['parent_id']
    #
    #         if parent_id in used_parent:
    #             continue
    #
    #         if parent_count_map[parent_id] > 2:
    #             print("Putting parent..")
    #             final_context.append(parent_chunks_df.loc[parent_id, 'text'])
    #             used_parent.add(parent_id)
    #         else:
    #             final_context.append(children_chunks_df.iloc[index]['text'])
    #
    #         if len(final_context) >= 5:
    #             break
    #
    #     return final_context
    #
    # def sentence_window_retrieval(self, query: str, children_chunks_df: pd.DataFrame, window_size: int):
    #     df_embeddings = pd.DataFrame(children_chunks_df['embeddings'].tolist())
    #     similarities = cosine_similarity(self.embedding_generator.get_embedding([query]), df_embeddings).flatten()
    #
    #     top_indices = np.argsort(similarities)[::-1][:5]
    #
    #     print(top_indices)
    #
    #     final_context = []
    #
    #     # parse from node to back, node to front using IDs. Make sure to keep track of order.
    #     for index in top_indices:
    #         cur_node = children_chunks_df.iloc[index]
    #
    #         prev_node = children_chunks_df.loc[cur_node['prev_id']]
    #         context_before = []
    #         while len(context_before) < window_size and prev_node.name != self.head_child_chunk.id:
    #             context_before.append(prev_node['text'])
    #             prev_node = children_chunks_df.loc[prev_node['prev_id']]
    #
    #         context_before.reverse()
    #
    #         next_node = children_chunks_df.loc[cur_node['next_id']]
    #         context_after = []
    #         while len(context_after) < window_size and next_node.name != self.tail_child_chunk.id:
    #             context_after.append(next_node['text'])
    #             next_node = children_chunks_df.loc[next_node['next_id']]
    #
    #         concat_context = ''.join(context_before) + cur_node.text + ''.join(context_after)
    #
    #         final_context.append(concat_context)
    #
    #     for context in final_context:
    #         print()
    #         print(context)
    #
    #     return final_context
