"""Ingestion entrypoint for loading documents and upserting to Pinecone."""

from __future__ import annotations

import argparse

from app.config import get_settings
from app.rag_pipeline import create_vector_store, load_catalog_documents, split_documents


def ingest_catalog(file_path: str) -> None:
    """Load catalog, split into chunks, embed, and upsert to Pinecone."""
    settings = get_settings()
    documents = load_catalog_documents(file_path)
    chunks = split_documents(documents)
    create_vector_store(chunks=chunks, settings=settings)
    print(f"Ingestion complete. Upserted {len(chunks)} chunks into Pinecone index.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest catalog into Pinecone.")
    parser.add_argument(
        "--file",
        default="catalog.txt",
        help="Path to catalog file (.txt or .pdf). Default: catalog.txt",
    )
    args = parser.parse_args()

    try:
        ingest_catalog(args.file)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Ingestion failed: {exc}")
        raise


if __name__ == "__main__":
    main()
