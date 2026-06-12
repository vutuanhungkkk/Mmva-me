"""Simple RAG builder: load documents, chunk, embed, and persist Chroma DB.

Usage:
  python -m assistant.rag_builder --source ./documents --persist ./vector_db

This script tries to import LangChain components but will fail fast with
instructions if dependencies are missing.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from .config import (
    RAG_DOCUMENTS_DIR,
    RAG_VECTOR_DB_DIR,
    RAG_CHUNK_SIZE,
    RAG_CHUNK_OVERLAP,
)


def _ensure_dir(d: str) -> str:
    p = Path(d)
    p.mkdir(parents=True, exist_ok=True)
    return str(p)


def build_vectorstore(source: str, persist_dir: str, chunk_size: int, chunk_overlap: int) -> None:
    try:
        from langchain_community.document_loaders import PyPDFLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from langchain_huggingface import HuggingFaceEmbeddings
        from langchain_chroma import Chroma
        from langchain_core.documents import Document

        src = Path(source)
        if not src.exists():
            raise SystemExit(f"Source directory not found: {source}")

        pdfs = list(src.rglob("*.pdf"))
        if not pdfs:
            raise SystemExit(f"No PDF files found in {source}")

        docs: List[Document] = []
        for p in pdfs:
            loader = PyPDFLoader(str(p))
            loaded = loader.load()
            docs.extend(loaded)

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        chunks = splitter.split_documents(docs)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
        )

        persist_dir = _ensure_dir(persist_dir)
        vect = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_dir)
        try:
            count = getattr(vect, "_collection").count()
        except Exception:
            count = len(chunks)

        print(f"Persisted vectorstore to {persist_dir} with ~{count} chunks")
        try:
            vect.persist()
        except Exception:
            pass

    except Exception as exc:
        print("Error building vectorstore:", exc)
        print("Install required packages: pip install langchain chromadb sentence-transformers pypdf")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=RAG_DOCUMENTS_DIR)
    parser.add_argument("--persist", default=RAG_VECTOR_DB_DIR)
    parser.add_argument("--chunk-size", type=int, default=RAG_CHUNK_SIZE)
    parser.add_argument("--chunk-overlap", type=int, default=RAG_CHUNK_OVERLAP)
    args = parser.parse_args()

    build_vectorstore(args.source, args.persist, args.chunk_size, args.chunk_overlap)


if __name__ == "__main__":
    main()
