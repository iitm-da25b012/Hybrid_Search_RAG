from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    """
    Represents a single text chunk.
    """

    chunk_id: int
    text: str
    metadata: dict


class FixedChunker:
    """
    Fixed-size chunker with overlap.

    Example:
        chunk_size = 500
        overlap = 100

        Chunk 1 -> 0-500
        Chunk 2 -> 400-900
        Chunk 3 -> 800-1300
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 100,
    ):
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        if overlap >= chunk_size:
            raise ValueError(
                "overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_documents(self, documents: List) -> List[Chunk]:
        """
        Chunk a list of Document objects.

        Parameters
        ----------
        documents : List[Document]

        Returns
        -------
        List[Chunk]
        """

        chunks = []
        chunk_id = 0

        for document in documents:

            text = document.text
            metadata = document.metadata

            start = 0

            while start < len(text):

                end = min(start + self.chunk_size, len(text))

                chunk_text = text[start:end]

                chunk_metadata = metadata.copy()

                chunk_metadata.update(
                    {
                        "chunk_id": chunk_id,
                        "chunk_index": chunk_id,
                        "chunk_strategy": "fixed",
                        "character_count": len(chunk_text),
                        "start_char": start,
                        "end_char": end,
                    }
                )

                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        metadata=chunk_metadata,
                    )
                )

                chunk_id += 1

                if end == len(text):
                    break

                start = end - self.overlap

        return chunks