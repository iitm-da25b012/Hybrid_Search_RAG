from src.embeddings.embedder import Embedder

embedder = Embedder()

vector = embedder.embed(
    "Machine Learning is awesome."
)

print(len(vector))

print(vector[:10])