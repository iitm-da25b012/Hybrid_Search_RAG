from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi


class BM25Index:
    """
    BM25 keyword index for Hybrid Search RAG.

    Stores:
        - BM25 index
        - Original chunks
        - Chunk metadata

    Can be saved and reloaded from disk.
    """

    def __init__(
        self,
        save_directory: str = "data/embeddings/bm25",
    ):

        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.index = None
        self.documents = []
        self.metadata = []

    # -----------------------------------------------------

    @staticmethod
    def tokenize(
        text: str,
    ) -> List[str]:
        """
        Basic tokenizer.

        Can later be replaced by SpaCy or NLTK.
        """

        return text.lower().split()

    # -----------------------------------------------------

    def build(
        self,
        chunks,
    ) -> None:
        """
        Build BM25 index from chunks.
        """

        self.documents = [
            chunk.text
            for chunk in chunks
        ]

        self.metadata = [
            chunk.metadata
            for chunk in chunks
        ]

        tokenized = [
            self.tokenize(doc)
            for doc in self.documents
        ]

        self.index = BM25Okapi(tokenized)

    # -----------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Search using BM25.
        """

        if self.index is None:
            raise RuntimeError(
                "BM25 index has not been built."
            )

        tokens = self.tokenize(query)

        scores = self.index.get_scores(tokens)

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k]

        results = []

        for idx, score in ranked:

            results.append(
                {
                    "score": float(score),
                    "document": self.documents[idx],
                    "metadata": self.metadata[idx],
                }
            )

        return results

    # -----------------------------------------------------

    def save(
        self,
    ) -> None:
        """
        Save BM25 index.
        """

        with open(
            self.save_directory / "bm25.pkl",
            "wb",
        ) as f:

            pickle.dump(
                {
                    "index": self.index,
                    "documents": self.documents,
                    "metadata": self.metadata,
                },
                f,
            )

    # -----------------------------------------------------

    def load(
        self,
    ) -> None:
        """
        Load BM25 index.
        """

        file = self.save_directory / "bm25.pkl"

        if not file.exists():
            raise FileNotFoundError(
                "BM25 index not found."
            )

        with open(
            file,
            "rb",
        ) as f:

            data = pickle.load(f)

        self.index = data["index"]
        self.documents = data["documents"]
        self.metadata = data["metadata"]

    # -----------------------------------------------------

    def count(
        self,
    ) -> int:
        """
        Number of indexed chunks.
        """

        return len(self.documents)

    # -----------------------------------------------------

    def reset(
        self,
    ) -> None:
        """
        Clear the index.
        """

        self.index = None
        self.documents = []
        self.metadata = []

        file = self.save_directory / "bm25.pkl"

        if file.exists():
            file.unlink()