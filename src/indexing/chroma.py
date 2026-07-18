from __future__ import annotations

from typing import List, Dict, Any

import chromadb
from chromadb.config import Settings


class ChromaIndex:
    """
    Wrapper around ChromaDB for storing and retrieving
    document embeddings.
    """

    def __init__(
        self,
        persist_directory: str = "data/embeddings/chroma",
        collection_name: str = "hybrid-search",
    ):

        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False
            ),
        )

        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={
                "description": "Hybrid Search RAG"
            },
        )

    # -----------------------------------------------------

    def add(
        self,
        chunks,
        embeddings: List[List[float]],
    ) -> None:
        """
        Store chunks and embeddings inside ChromaDB.
        """

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Chunks and embeddings must have equal length."
            )

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:

            metadata = dict(chunk.metadata)

            metadata["character_count"] = len(chunk.text)

            ids.append(
                f'{metadata["document_id"]}_{metadata["chunk_id"]}'
            )

            documents.append(
                chunk.text
            )

            metadatas.append(metadata)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    # -----------------------------------------------------

    def search(
        self,
        embedding: List[float],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Vector similarity search.
        """

        return self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
        )

    # -----------------------------------------------------

    def get(
        self,
        chunk_id: str,
    ) -> Dict[str, Any]:
        """
        Retrieve one chunk.
        """

        return self.collection.get(
            ids=[chunk_id]
        )

    # -----------------------------------------------------

    def delete(
        self,
        chunk_ids: List[str],
    ) -> None:
        """
        Delete chunks.
        """

        self.collection.delete(
            ids=chunk_ids
        )

    # -----------------------------------------------------

    def count(
        self,
    ) -> int:
        """
        Number of indexed chunks.
        """

        return self.collection.count()

    # -----------------------------------------------------

    def reset(
        self,
    ) -> None:
        """
        Remove every chunk.
        """

        ids = self.collection.get()["ids"]

        if ids:
            self.collection.delete(ids=ids)

    # -----------------------------------------------------

    def peek(
        self,
        limit: int = 5,
    ):
        """
        View a few stored chunks.
        """

        return self.collection.peek(limit=limit)

    # -----------------------------------------------------

    def exists(
        self,
        chunk_id: str,
    ) -> bool:

        result = self.collection.get(
            ids=[chunk_id]
        )

        return len(result["ids"]) > 0