from __future__ import annotations

from typing import List, Dict, Any

from src.embeddings.embedder import Embedder
from src.indexing.chroma import ChromaIndex
from src.indexing.bm25 import BM25Index


class HybridRetriever:
    """
    Combines dense retrieval (ChromaDB)
    and sparse retrieval (BM25).

    Uses weighted score fusion.
    """

    def __init__(
        self,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ):

        if dense_weight + sparse_weight != 1.0:
            raise ValueError(
                "Weights must add up to 1."
            )

        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        self.embedder = Embedder()

        self.chroma = ChromaIndex()

        self.bm25 = BM25Index()

        try:
            self.bm25.load()
            print("BM25 index loaded")
        except FileNotFoundError:
            print("BM25 index not found. Run indexing first.")


    # --------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Hybrid search.

        Returns ranked results from both indexes.
        """

        dense_results = self._dense_search(
            query,
            top_k,
        )

        sparse_results = self._sparse_search(
            query,
            top_k,
        )


        fused_results = self._fuse_results(
            dense_results,
            sparse_results,
        )


        return fused_results[:top_k]


    # --------------------------------------------------

    def _dense_search(
        self,
        query: str,
        top_k: int,
    ):

        embedding = self.embedder.embed(
            query
        )

        results = self.chroma.search(
            embedding,
            top_k,
        )


        output = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]


        for doc, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):

            output.append(
                {
                    "text": doc,
                    "metadata": metadata,
                    "score": self._normalize_distance(
                        distance
                    ),
                    "source": "dense",
                }
            )


        return output


    # --------------------------------------------------

    def _sparse_search(
        self,
        query: str,
        top_k: int,
    ):

        results = self.bm25.search(
            query,
            top_k,
        )


        scores = [
            result["score"]
            for result in results
        ]


        normalized = self._normalize_scores(
            scores
        )


        output = []


        for result, score in zip(
            results,
            normalized,
        ):

            output.append(
                {
                    "text": result["document"],
                    "metadata": result["metadata"],
                    "score": score,
                    "source": "sparse",
                }
            )


        return output


    # --------------------------------------------------

    def _fuse_results(
        self,
        dense_results,
        sparse_results,
    ):

        combined = {}


        # Dense scores
        for result in dense_results:

            chunk_id = result["metadata"]["chunk_id"]

            combined[chunk_id] = {
                "text": result["text"],
                "metadata": result["metadata"],
                "dense_score": result["score"],
                "sparse_score": 0,
            }


        # Sparse scores
        for result in sparse_results:

            chunk_id = result["metadata"]["chunk_id"]


            if chunk_id in combined:

                combined[chunk_id]["sparse_score"] = (
                    result["score"]
                )

            else:

                combined[chunk_id] = {
                    "text": result["text"],
                    "metadata": result["metadata"],
                    "dense_score": 0,
                    "sparse_score": result["score"],
                }


        final_results = []


        for item in combined.values():

            final_score = (
                self.dense_weight
                * item["dense_score"]
                +
                self.sparse_weight
                * item["sparse_score"]
            )


            item["hybrid_score"] = final_score

            final_results.append(item)


        final_results.sort(
            key=lambda x: x["hybrid_score"],
            reverse=True,
        )


        return final_results


    # --------------------------------------------------

    @staticmethod
    def _normalize_distance(
        distance: float,
    ) -> float:
        """
        Convert Chroma distance into similarity score.

        Lower distance = better.
        """

        return 1 / (1 + distance)


    # --------------------------------------------------

    @staticmethod
    def _normalize_scores(
        scores,
    ):

        if not scores:
            return []


        minimum = min(scores)
        maximum = max(scores)


        if maximum == minimum:
            return [
                1.0
                for _ in scores
            ]


        return [
            (score - minimum)
            /
            (maximum - minimum)
            for score in scores
        ]