from pathlib import Path

from src.ingestion.loader import DocumentLoader

loader = DocumentLoader()

test_folder = Path("test")

for file in test_folder.iterdir():

    if file.is_file():

        print("=" * 70)
        print(f"Loading {file.name}")

        documents = loader.load(file)

        print(f"Documents created : {len(documents)}")

        for doc in documents:

            print(doc.metadata)

            print(doc.text[:150])

            print("-" * 40)