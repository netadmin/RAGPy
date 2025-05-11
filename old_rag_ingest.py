#!/usr/bin/env python3
import os
import sys
import argparse
import torch

from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document

def clean_metadata(raw_meta: dict) -> dict:
    """Keep only primitive metadata values."""
    return {k: v for k, v in raw_meta.items() if isinstance(v, (str, int, float, bool))}

def ingest(library_path: str):
    output_path = os.path.join(library_path, "vector_index_chroma")
    os.makedirs(output_path, exist_ok=True)

    supported_exts = {".pdf", ".txt", ".docx", ".py"}
    docs: list[Document] = []

    print(f"📂 Scanning {library_path}")
    for root, _, files in os.walk(library_path):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in supported_exts:
                continue
            path = os.path.join(root, fn)
            print(f"  ➕ Loading {path}")
            try:
                loader = UnstructuredLoader(file_path=path)
                raw = loader.load()
            except Exception as e:
                print(f"    ❌ Load failed: {e}")
                continue

            count = 0
            for item in raw:
                if isinstance(item, Document):
                    content, meta = item.page_content, item.metadata
                elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict):
                    content, meta = item
                else:
                    continue
                meta = clean_metadata(meta)
                docs.append(Document(page_content=content, metadata=meta))
                count += 1
            print(f"    ✅ Queued {count} pieces from {fn}")

    if not docs:
        print("⚠️ No documents found—exiting.")
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    print(f"✂️ {len(chunks)} chunks generated")
    if not chunks:
        print("⚠️ No chunks—exiting.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🤖 Embedding on {device}")
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": device}
    )

    print(f"💾 Initializing Chroma store at {output_path}")
    store = Chroma(persist_directory=output_path, embedding_function=embedder)
    print(f"➕ Adding {len(chunks)} chunks to the index")
    
    BATCH_SIZE = 5000  # safely under the ~5461 limit

    print(f"➕ Adding {len(chunks)} chunks in batches of {BATCH_SIZE}")
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        print(f"  • Batch {i//BATCH_SIZE + 1}: {len(batch)} docs")
        store.add_documents(batch)

    print("✅ Chroma index written.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest docs into a Chroma RAG index")
    parser.add_argument("-l", "--library-path", required=True,
                        help="Root folder of your docs (PDF, DOCX, TXT, PY)")
    args = parser.parse_args()

    if not os.path.isdir(args.library_path):
        print(f"❌ Not a directory: {args.library_path}")
        sys.exit(1)

    ingest(args.library_path)
