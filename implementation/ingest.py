"""Build the Chroma index from the company PDFs in knowledge-files/."""

from pathlib import Path

from chromadb import PersistentClient

from implementation.embeddings import embed_texts
from implementation.preprocess import build_chunks


DB_NAME = str(Path(__file__).parent.parent / "preprocessed_db")
collection_name = "docs"


def create_embeddings(chunks):
    chroma = PersistentClient(path=DB_NAME)
    if collection_name in [collection.name for collection in chroma.list_collections()]:
        chroma.delete_collection(collection_name)

    texts = [chunk.page_content for chunk in chunks]
    vectors = embed_texts(texts, "document")
    collection = chroma.get_or_create_collection(collection_name)
    collection.add(
        ids=[str(index) for index in range(len(chunks))],
        embeddings=vectors,
        documents=texts,
        metadatas=[chunk.metadata for chunk in chunks],
    )
    print(f"Vectorstore created with {collection.count()} documents", flush=True)


if __name__ == "__main__":
    documents = build_chunks()
    create_embeddings(documents)
    print("Ingestion complete")
