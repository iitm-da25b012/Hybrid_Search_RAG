from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List


@dataclass
class Chunk:
    chunk_id: int
    text: str
    metadata: dict


class RecursiveChunker:
    """
    Structure-aware recursive chunker.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 100,
    ):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_documents(self, documents: List) -> List[Chunk]:

        chunks = []
        chunk_id = 0

        for document in documents:

            sections = self._split_by_headers(document.text)

            for section in sections:

                pieces = self._recursive_split(section)

                for piece in pieces:

                    metadata = document.metadata.copy()

                    metadata.update(
                        {
                            "chunk_id": chunk_id,
                            "chunk_strategy": "recursive",
                        }
                    )

                    chunks.append(
                        Chunk(
                            chunk_id=chunk_id,
                            text=piece,
                            metadata=metadata,
                        )
                    )

                    chunk_id += 1

        return chunks

    # ----------------------------------------------------
    # Recursive splitter
    # ----------------------------------------------------

    def _recursive_split(self, text: str) -> List[str]:

        text = text.strip()

        if len(text) <= self.chunk_size:
            return [text]

        paragraphs = text.split("\n\n")

        if len(paragraphs) > 1:
            return self._merge_units(paragraphs)

        sentences = re.split(r'(?<=[.!?])\s+', text)

        if len(sentences) > 1:
            return self._merge_units(sentences)

        return self._fixed_split(text)

    # ----------------------------------------------------
    # Merge small units into chunks
    # ----------------------------------------------------

    def _merge_units(self, units: List[str]) -> List[str]:

        chunks = []

        current = ""

        for unit in units:

            candidate = current + "\n\n" + unit if current else unit

            if len(candidate) <= self.chunk_size:

                current = candidate

            else:

                if current:
                    chunks.append(current)

                current = unit

        if current:
            chunks.append(current)

        return chunks

    # ----------------------------------------------------
    # Markdown headers
    # ----------------------------------------------------

    def _split_by_headers(self, text: str) -> List[str]:

        pattern = r'(?=^#{1,6}\s)'

        sections = re.split(
            pattern,
            text,
            flags=re.MULTILINE,
        )

        sections = [s.strip() for s in sections if s.strip()]

        if sections:
            return sections

        return [text]

    # ----------------------------------------------------
    # Fallback
    # ----------------------------------------------------

    def _fixed_split(self, text: str) -> List[str]:

        chunks = []

        start = 0

        while start < len(text):

            end = min(start + self.chunk_size, len(text))

            chunks.append(text[start:end])

            if end == len(text):
                break

            start = end - self.overlap

        return chunks