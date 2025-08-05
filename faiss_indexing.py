import faiss
import numpy as np
import json


def build_faiss_index(
    embedding_path='chunk_embeddings.npy',
    metadata_path='chunked_output.json',
    index_path='faiss_index.bin'
):
    # Load embeddings
    embeddings = np.load(embedding_path).astype('float32')  # FAISS requires float32

    # Normalize embeddings to unit length for cosine similarity (recommended)
    faiss.normalize_L2(embeddings)

    # Load chunk metadata
    with open(metadata_path, 'r', encoding='utf-8') as f:
        chunk_metadata = json.load(f)

    dimension = embeddings.shape[1]

    # Create FAISS index for Inner Product (IP)
    index = faiss.IndexFlatIP(dimension)

    # Add normalized embeddings to index
    index.add(embeddings)

    print(f"Indexed {index.ntotal} vectors with dimension {dimension}.")

    # Save FAISS index to disk
    faiss.write_index(index, index_path)
    print(f"FAISS index saved to '{index_path}'.")

    return index, chunk_metadata


if __name__ == "__main__":
    build_faiss_index()
