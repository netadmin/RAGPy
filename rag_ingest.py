#!/usr/bin/env python3
import os
import sys
import argparse
import torch
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional

from tqdm import tqdm
from langchain_unstructured import UnstructuredLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter
)
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document


def clean_metadata(raw_meta: dict) -> dict:
    """Keep only primitive metadata values."""
    return {k: v for k, v in raw_meta.items() if isinstance(v, (str, int, float, bool))}


def get_last_processed_time(tracking_file: str) -> Optional[float]:
    """Get timestamp of last processing run."""
    if os.path.exists(tracking_file):
        with open(tracking_file, "r") as f:
            try:
                return float(f.read().strip())
            except:
                return None
    return None


def save_processed_time(tracking_file: str) -> None:
    """Save current timestamp for incremental processing."""
    with open(tracking_file, "w") as f:
        f.write(str(datetime.now().timestamp()))


def hash_content(text: str) -> str:
    """Create a hash of content for deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def ingest(
    library_path: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    incremental: bool = False,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    output_path = os.path.join(library_path, "vector_index_chroma")
    os.makedirs(output_path, exist_ok=True)
    
    tracking_file = os.path.join(output_path, ".last_process_time")
    last_process_time = None
    
    if incremental:
        last_process_time = get_last_processed_time(tracking_file)
        print(f"🔄 Incremental mode: only processing files modified after {datetime.fromtimestamp(last_process_time).strftime('%Y-%m-%d %H:%M:%S')}" if last_process_time else "🔄 Incremental mode: no previous run found, processing all files")

    supported_exts = {".pdf", ".txt", ".docx", ".py", ".md", ".html"}
    docs: List[Document] = []

    # Scan for files
    all_files = []
    print(f"📂 Scanning {library_path}")
    for root, _, files in os.walk(library_path):
        for fn in files:
            ext = os.path.splitext(fn)[1].lower()
            if ext not in supported_exts:
                continue
            path = os.path.join(root, fn)
            
            # Skip if in incremental mode and file is not new/modified
            if incremental and last_process_time:
                mtime = os.path.getmtime(path)
                if mtime <= last_process_time:
                    continue
                    
            all_files.append(path)
    
    # Process files with progress bar
    for path in tqdm(all_files, desc="Loading documents"):
        fn = os.path.basename(path)
        try:
            loader = UnstructuredLoader(file_path=path)
            raw = loader.load()
        except Exception as e:
            print(f"    ❌ Loading {fn} failed: {e}")
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
            meta["source"] = path  # Ensure source path is always included
            docs.append(Document(page_content=content, metadata=meta))
            count += 1
        if count > 0:
            print(f"    ✅ Queued {count} pieces from {fn}")

    if not docs:
        print("⚠️ No documents found—exiting.")
        return

    # Choose appropriate text splitters based on document types
    print("✂️ Splitting documents into chunks")
    ext_to_splitter = {
        ".md": MarkdownHeaderTextSplitter(headers_to_split_on=[
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]),
        # Add more specialized splitters as needed
    }
    
    # Default splitter for all document types
    default_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    
    # Process each document with appropriate splitter
    chunks = []
    for doc in tqdm(docs, desc="Chunking documents"):
        doc_ext = os.path.splitext(doc.metadata.get("source", ""))[1].lower()
        specialized_splitter = ext_to_splitter.get(doc_ext)
        
        if specialized_splitter and isinstance(specialized_splitter, MarkdownHeaderTextSplitter):
            # For markdown, first split by headers then by chars
            header_splits = specialized_splitter.split_text(doc.page_content)
            for split in header_splits:
                # Add header info to metadata
                header_meta = {**doc.metadata}
                for key, val in split.metadata.items():
                    header_meta[key] = val
                # Further split by chars if needed
                char_chunks = default_splitter.split_text(split.page_content)
                for chunk in char_chunks:
                    chunks.append(Document(page_content=chunk, metadata=header_meta))
        else:
            # Use default char splitter for all other docs
            doc_chunks = default_splitter.split_documents([doc])
            chunks.extend(doc_chunks)
    
    print(f"✂️ {len(chunks)} chunks generated")
    if not chunks:
        print("⚠️ No chunks—exiting.")
        return

    # Deduplicate chunks
    print("🧹 Removing duplicate chunks...")
    seen_contents = set()
    unique_chunks = []
    for chunk in tqdm(chunks, desc="Deduplicating"):
        content_hash = hash_content(chunk.page_content)
        if content_hash not in seen_contents:
            seen_contents.add(content_hash)
            unique_chunks.append(chunk)
    
    print(f"🧹 Removed {len(chunks) - len(unique_chunks)} duplicate chunks")
    chunks = unique_chunks

    # Set up embeddings
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🤖 Embedding on {device} using {model_name}")
    embedder = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": device}
    )

    # Initialize Chroma
    print(f"💾 Initializing Chroma store at {output_path}")
    store = Chroma(persist_directory=output_path, embedding_function=embedder)
    
    # Add documents in batches
    BATCH_SIZE = 5000  # safely under the ~5461 limit
    print(f"➕ Adding {len(chunks)} chunks in batches of {BATCH_SIZE}")
    
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        print(f"  • Batch {i//BATCH_SIZE + 1}: {len(batch)} docs")
        store.add_documents(batch)

    print("✅ Chroma index written.")
    
    # Save process timestamp for incremental updates
    if incremental:
        save_processed_time(tracking_file)
        print("🕒 Saved processing timestamp for future incremental updates")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest docs into a Chroma RAG index")
    parser.add_argument(
        "-l", "--library-path", required=True,
        help="Root folder of your docs (PDF, DOCX, TXT, PY, MD, HTML)"
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500,
        help="Size of text chunks for embedding"
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=100,
        help="Overlap between chunks"
    )
    parser.add_argument(
        "--incremental", action="store_true",
        help="Only process new or modified files since last run"
    )
    parser.add_argument(
        "--model", default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model to use"
    )
    
    args = parser.parse_args()

    if not os.path.isdir(args.library_path):
        print(f"❌ Not a directory: {args.library_path}")
        sys.exit(1)

    ingest(
        library_path=args.library_path,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        incremental=args.incremental,
        model_name=args.model,
    )