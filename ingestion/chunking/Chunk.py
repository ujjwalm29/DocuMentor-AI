from dataclasses import dataclass
from typing import List


@dataclass
class ChunkBase:

    id: int
    text: str
    prev_id: int
    next_id: int
    embeddings: List[float]
    metadata: object


@dataclass
class ParentChunk(ChunkBase):

    number_of_children: int


@dataclass
class ChildChunk(ChunkBase):

    parent_id: int
