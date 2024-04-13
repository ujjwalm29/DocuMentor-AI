from dataclasses import dataclass
from typing import List
from uuid import UUID


@dataclass
class Chunk:

    chunk_id: UUID
    text: str
    prev_id: UUID | None
    next_id: UUID | None
    embeddings: List[float]
    metadata: object
    number_of_children: int
    parent_id: UUID | None