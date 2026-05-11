"""
Knowledge Base Ingestion Script
================================
One-time script to read all documents from knowledge_base/,
chunk them, compute TF-IDF embeddings, and store in a ChromaDB collection.

Uses sklearn TF-IDF (fully offline, no model downloads needed).

Usage:
    python3 ingest_knowledge.py
"""

import os
import glob
import json
import pickle
import numpy as np
import chromadb
from sklearn.feature_extraction.text import TfidfVectorizer

# ── Configuration ───────────────────────────────────────────────────────
KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(__file__), "knowledge_base")
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
TFIDF_MODEL_PATH = os.path.join(os.path.dirname(__file__), "chroma_db", "tfidf_model.pkl")
COLLECTION_NAME = "triage_knowledge"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 80
FILE_EXTENSIONS = [".md", ".txt"]
EMBEDDING_DIM = 384  # output embedding dimension


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into overlapping chunks respecting paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = ""

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk += ("\n\n" + para) if current_chunk else para
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            if len(para) > chunk_size:
                words = para.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) + 1 <= chunk_size:
                        current_chunk += (" " + word) if current_chunk else word
                    else:
                        chunks.append(current_chunk.strip())
                        overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                        current_chunk = overlap_text + " " + word
            else:
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                current_chunk = overlap_text + "\n\n" + para if overlap_text else para

    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    return chunks


def load_documents(base_dir):
    """Recursively load all documents from the knowledge base directory."""
    documents = []
    for ext in FILE_EXTENSIONS:
        pattern = os.path.join(base_dir, "**", f"*{ext}")
        for filepath in glob.glob(pattern, recursive=True):
            rel_path = os.path.relpath(filepath, base_dir)
            parts = rel_path.split(os.sep)
            category = parts[0] if len(parts) > 1 else "general"
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append({
                "content": content,
                "source": rel_path,
                "filename": os.path.basename(filepath),
                "category": category,
            })
            print(f"  Loaded: {rel_path} ({len(content)} chars)")
    return documents


def tfidf_embed(texts, fit=True, vectorizer=None):
    """
    Compute TF-IDF embeddings, then reduce/pad to EMBEDDING_DIM.
    Returns (embeddings as list of lists, fitted vectorizer).
    """
    if fit or vectorizer is None:
        vectorizer = TfidfVectorizer(
            max_features=EMBEDDING_DIM,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(texts)
    else:
        matrix = vectorizer.transform(texts)

    embeddings = matrix.toarray().tolist()
    # Pad to EMBEDDING_DIM if fewer features
    for i in range(len(embeddings)):
        if len(embeddings[i]) < EMBEDDING_DIM:
            embeddings[i] += [0.0] * (EMBEDDING_DIM - len(embeddings[i]))

    return embeddings, vectorizer


def ingest():
    """Main ingestion pipeline."""
    print("=" * 60)
    print("Knowledge Base Ingestion (TF-IDF Offline Embeddings)")
    print("=" * 60)

    # ── Load documents ──
    print(f"\n📂 Loading documents from: {KNOWLEDGE_BASE_DIR}")
    documents = load_documents(KNOWLEDGE_BASE_DIR)
    print(f"   Loaded {len(documents)} documents")

    if not documents:
        print("❌ No documents found.")
        return

    # ── Chunk documents ──
    print(f"\n✂️  Chunking documents (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    all_chunks = []
    all_metadatas = []
    all_ids = []
    chunk_counter = 0

    for doc in documents:
        chunks = chunk_text(doc["content"])
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": doc["source"],
                "filename": doc["filename"],
                "category": doc["category"],
                "chunk_index": i,
                "total_chunks": len(chunks),
            })
            all_ids.append(f"chunk_{chunk_counter}")
            chunk_counter += 1

    print(f"   Created {len(all_chunks)} chunks from {len(documents)} documents")

    # ── Compute TF-IDF embeddings ──
    print(f"\n🔢 Computing TF-IDF embeddings (dim={EMBEDDING_DIM})...")
    embeddings, vectorizer = tfidf_embed(all_chunks, fit=True)
    print(f"   Computed {len(embeddings)} embeddings")

    # Save vectorizer for query-time use
    os.makedirs(os.path.dirname(TFIDF_MODEL_PATH), exist_ok=True)
    with open(TFIDF_MODEL_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    print(f"   Saved TF-IDF model to {TFIDF_MODEL_PATH}")

    # ── Initialize ChromaDB ──
    print(f"\n💾 Initializing ChromaDB at: {CHROMA_PERSIST_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    try:
        client.delete_collection(COLLECTION_NAME)
        print(f"   Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "Fall detection triage knowledge base", "hnsw:space": "cosine"},
    )
    print(f"   Created collection '{COLLECTION_NAME}'")

    # ── Store in ChromaDB ──
    print(f"\n📥 Storing {len(all_chunks)} chunks...")
    batch_size = 50
    for i in range(0, len(all_chunks), batch_size):
        end = min(i + batch_size, len(all_chunks))
        collection.add(
            documents=all_chunks[i:end],
            embeddings=embeddings[i:end],
            metadatas=all_metadatas[i:end],
            ids=all_ids[i:end],
        )
        print(f"   Stored batch {i // batch_size + 1}/{(len(all_chunks) - 1) // batch_size + 1}")

    # ── Verify ──
    count = collection.count()
    print(f"\n✅ Ingestion complete! Collection '{COLLECTION_NAME}' has {count} chunks.")

    # Test query
    print("\n🔍 Test query: 'lateral fall hip fracture triage'")
    test_emb, _ = tfidf_embed(["lateral fall hip fracture triage"], fit=False, vectorizer=vectorizer)
    results = collection.query(query_embeddings=test_emb, n_results=3)
    for i, (doc, meta) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
        print(f"   [{i+1}] Source: {meta['source']}")
        print(f"       Preview: {doc[:120]}...")
        print()

    print("=" * 60)
    print("Done. You can now run the RAG pipeline.")
    print("=" * 60)


if __name__ == "__main__":
    ingest()
