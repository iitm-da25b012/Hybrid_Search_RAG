from src.ingestion.loader import DocumentLoader
from src.chunking.manager import ChunkManager
from src.indexing.sync import SyncManager


def main():

    loader = DocumentLoader()

    documents = loader.load(
        "test/Markdown.md"
    )

    print(f"Loaded {len(documents)} document(s)")

    manager = ChunkManager()

    chunks = manager.chunk_documents(
        documents,
        output_name="markdown"
    )

    print(f"Generated {len(chunks)} chunks")

    sync = SyncManager()

    sync.index(chunks)

    sync.verify()

    print(sync.statistics())


if __name__ == "__main__":
    main()