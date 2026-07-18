from src.ingestion.loader import DocumentLoader
from src.chunking.fixed import FixedChunker

loader = DocumentLoader()
documents = loader.load("test/Markdown.md")

chunker = FixedChunker(
    chunk_size=100,
    overlap=20
)

chunks = chunker.chunk_documents(documents)

print(f"Total chunks: {len(chunks)}")

for chunk in chunks:
    print("=" * 60)
    print(chunk.metadata)
    print(chunk.text)