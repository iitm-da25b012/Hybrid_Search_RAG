from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from src.embeddings.embedder import Embedder


@dataclass
class Chunk:
    chunk_id: int
    text: str
    metadata: dict


class SemanticChunker:
    """
    Semantic chunker that groups paragraphs based on embedding similarity.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.80,
        min_chunk_size: int = 300,
        max_chunk_size: int = 800,
    ):
        self.similarity_threshold = similarity_threshold
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

        self.embedder = Embedder()

    def chunk_documents(self, documents: List) -> List[Chunk]:

        chunks = []
        chunk_id = 0

        for document in documents:

            paragraphs = self._split_into_paragraphs(document.text)

            if not paragraphs:
                continue

            if len(paragraphs) == 1:

                metadata = document.metadata.copy()

                metadata.update(
                    {
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_id,
                        "chunk_strategy": "semantic",
                        "character_count": len(current_chunk),
                    }
                )

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=paragraphs[0],
                        metadata=metadata,
                    )
                )

                chunk_id += 1
                continue

            embeddings = self.embedder.embed_batch(paragraphs)

            current_chunk = paragraphs[0]

            for i in range(1, len(paragraphs)):

                similarity = self.cosine_similarity(
                    embeddings[i - 1],
                    embeddings[i],
                )

                candidate = current_chunk + "\n\n" + paragraphs[i]

                # Current chunk is too small -> keep growing
                if len(current_chunk) < self.min_chunk_size:
                    current_chunk = candidate
                    continue

                # Same topic and under max size
                if (
                    similarity >= self.similarity_threshold
                    and len(candidate) <= self.max_chunk_size
                ):
                    current_chunk = candidate

                else:

                    metadata = document.metadata.copy()

                    metadata.update(
                        {
                            "chunk_id": chunk_id,
                            "chunk_index": chunk_id,
                            "chunk_strategy": "semantic",
                            "character_count": len(current_chunk),
                        }
                    )
                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            text=current_chunk.strip(),
                            metadata=metadata,
                        )
                    )

                    chunk_id += 1
                    current_chunk = paragraphs[i]

            if current_chunk.strip():

                metadata = document.metadata.copy()

                metadata.update(
                    {
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_id,
                        "chunk_strategy": "semantic",
                        "character_count": len(current_chunk),
                    }
                )

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=current_chunk.strip(),
                        metadata=metadata,
                    )
                )

                chunk_id += 1

        return chunks

    # ----------------------------------------------------

    @staticmethod
    def _split_into_paragraphs(text: str) -> List[str]:

        paragraphs = [
            p.strip()
            for p in text.split("\n\n")
            if p.strip()
        ]

        return paragraphs

    # ----------------------------------------------------

    @staticmethod
    def cosine_similarity(a, b) -> float:

        a = np.array(a)
        b = np.array(b)

        denominator = np.linalg.norm(a) * np.linalg.norm(b)

        if denominator == 0:
            return 0.0

        return float(np.dot(a, b) / denominator)