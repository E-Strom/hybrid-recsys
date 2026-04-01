# generate_embeddings.py
import os
import argparse
import pickle
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def prepare_prompt(row):
    # Minimal fields: title + genres + synopsis
    return f"Title: {row['Name']}. Genres: {row['Genres']}. Synopsis: {row['Synopsis']}."

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load anime CSV
    anime_data = pd.read_csv(args.anime_csv)
    anime_data = anime_data[anime_data['Type'].isin(['TV', 'Movie'])]  # filter

    # Get anime_ids and prompts
    anime_ids = anime_data['anime_id'].tolist()
    assert len(anime_ids) == len(set(anime_ids)), "Duplicate anime_id detected!"

    # Build mapping: anime_id -> embedding index
    anime_id_to_idx = {anime_id: idx for idx, anime_id in enumerate(anime_ids)}
    
    # Prepare prompts
    prompts = anime_data.apply(prepare_prompt, axis=1).tolist()

    # Load model
    print(f"Loading model {args.model_name}...")
    model = SentenceTransformer(args.model_name)
    model = model.to(device)

    # Generate embeddings
    print("Generating embeddings...")
    embeddings = model.encode(
        prompts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        device=device
    )

    embeddings = np.array(embeddings)
    embedding_dim = embeddings.shape[1]

    # Save everything
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump({
            "embeddings": embeddings,
            "anime_ids": anime_ids,
            "anime_id_to_idx": anime_id_to_idx,
            "embedding_dim": embedding_dim,
            "model_name": args.model_name
        }, f)

    print(f"Saved embeddings ({embedding_dim} dim) to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate anime embeddings with LLM")
    parser.add_argument("--anime_csv", type=str, required=True, help="Path to anime CSV file")
    parser.add_argument("--output_path", type=str, default="../../data/processed/attentionrec/anime-embeddings.pkl", help="Path to save embeddings")
    parser.add_argument("--model_name", type=str, default="stsb-roberta-large", help="Sentence Transformer model name")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for encoding")
    parser.add_argument("--device", type=str, default="cuda", help="Device: cuda or cpu")

    args = parser.parse_args()
    main(args)