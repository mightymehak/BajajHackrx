import json
import numpy as np
from sentence_transformers import SentenceTransformer

def generate_and_save_embeddings(chunks, embedding_model_name='all-MiniLM-L6-v2'):
    """
    Generate vector embeddings for a list of text chunks and save embeddings + metadata to disk.
    
    Args:
        chunks (list of dict): List of chunk dictionaries with at least 'chunk_text' field.
        embedding_model_name (str): SentenceTransformer model name.
    """
    model = SentenceTransformer(embedding_model_name)
    texts = [chunk['chunk_text'] for chunk in chunks]

    print(f"Generating embeddings for {len(texts)} chunks using model '{embedding_model_name}'...")
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Save embeddings as numpy array file
    np.save('chunk_embeddings.npy', embeddings)

    # Save chunk metadata with the embeddings
    with open("chunk_metadata.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    print(f"\nSaved embeddings to 'chunk_embeddings.npy' and metadata to 'chunk_metadata.json'.")
    print("Embeddings shape:", embeddings.shape)
    print("First embedding vector (rounded):", np.round(embeddings[0][:10], 3), "...")
