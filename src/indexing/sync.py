from __future__ import annotations

from typing import List

from src.embeddings.embedder import Embedder
from src.embeddings.deduplication import Deduplicator
from src.indexing.chroma import ChromaIndex
from src.indexing.bm25 import BM25Index


class SyncManager:
    """
    Keeps ChromaDB and BM25 synchronized.

    Pipeline:

        Chunks
           │
           ▼
      Generate Embeddings
           │
           ▼
      Deduplicate
           │
           ▼
      Update ChromaDB
           │
           ▼
      Update BM25

    Every chunk stored in ChromaDB also exists in BM25.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
    ):

        self.embedder = Embedder()

        self.deduplicator = Deduplicator(
            threshold=similarity_threshold
        )

        self.chroma = ChromaIndex()

        self.bm25 = BM25Index()

    # -----------------------------------------------------

    def index(
        self,
        chunks,
    ):
        """
        Complete indexing pipeline.
        """

        if len(chunks) == 0:
            return

        print(f"\nReceived {len(chunks)} chunks")

        # -------------------------------------------
        # Generate embeddings
        # -------------------------------------------

        texts = [
            chunk.text
            for chunk in chunks
        ]

        embeddings = self.embedder.embed_batch(
            texts
        )

        print("Embeddings generated")

        # -------------------------------------------
        # Remove duplicates
        # -------------------------------------------

        chunks, embeddings = (
            self.deduplicator.deduplicate(
                chunks,
                embeddings,
            )
        )

        print(
            f"{len(chunks)} chunks remain after deduplication"
        )

        # -------------------------------------------
        # Update ChromaDB
        # -------------------------------------------

        self.chroma.add(
            chunks,
            embeddings,
        )

        print("ChromaDB updated")

        # -------------------------------------------
        # Update BM25
        # -------------------------------------------

        self.bm25.build(
            chunks,
        )

        self.bm25.save()

        print("BM25 updated")

    # -----------------------------------------------------

    def search_vector(
        self,
        query: str,
        top_k: int = 5,
    ):

        embedding = self.embedder.embed(
            query
        )

        return self.chroma.search(
            embedding,
            top_k,
        )

    # -----------------------------------------------------

    def search_keyword(
        self,
        query: str,
        top_k: int = 5,
    ):

        return self.bm25.search(
            query,
            top_k,
        )

    # -----------------------------------------------------

    def statistics(
        self,
    ):

        return {

            "vector_chunks":
                self.chroma.count(),

            "bm25_chunks":
                self.bm25.count(),

            "synchronized":
                self.chroma.count()
                == self.bm25.count(),
        }

    # -----------------------------------------------------

    def reset(self):

        self.chroma.reset()

        self.bm25.reset()

        print("Indexes cleared")

    # -----------------------------------------------------

    def verify(self):

        chroma = self.chroma.count()

        bm25 = self.bm25.count()

        if chroma != bm25:

            raise RuntimeError(
                f"Index mismatch! "
                f"Chroma={chroma}, "
                f"BM25={bm25}"
            )

        print(
            f"Indexes synchronized ({chroma} chunks)"
        )

    # -----------------------------------------------------

    def rebuild(
        self,
        chunks,
    ):
        """
        Delete both indexes and rebuild.
        """

        self.reset()

        self.index(
            chunks,
        )

        self.verify()