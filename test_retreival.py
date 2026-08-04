from src.retreival.hybrid import HybridRetriever


retriever = HybridRetriever(
    dense_weight=0.6,
    sparse_weight=0.4,
)


results = retriever.search(
    "What is machine learning?",
    top_k=5,
)


for result in results:

    print("="*50)

    print(
        "Score:",
        result["hybrid_score"]
    )

    print(
        result["metadata"]
    )

    print(
        result["text"][:300]
    )