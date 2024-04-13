from dataclasses import dataclass
from typing import List
from uuid import UUID


@dataclass
class ChunkBase:

    chunk_id: UUID
    text: str
    prev_id: UUID | None
    next_id: UUID | None
    embeddings: List[float]
    metadata: object


@dataclass
class ParentChunk(ChunkBase):

    number_of_children: int


@dataclass
class ChildChunk(ChunkBase):

    parent_id: UUID | None
