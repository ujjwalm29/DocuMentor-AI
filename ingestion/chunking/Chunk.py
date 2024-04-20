from dataclasses import dataclass
from datetime import time
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
    user_id: UUID
    document_id: UUID


@dataclass
class ParentChunk(ChunkBase):

    number_of_children: int


@dataclass
class ChildChunk(ChunkBase):

    parent_id: UUID | None
    score: int = 0


@dataclass
class Document:
    name: str
    document_id: UUID
    user_id: UUID
    added_at: time
