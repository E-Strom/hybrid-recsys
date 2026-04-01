import pandas as pd
import pickle
from pathlib import Path


# Paths
# Original relative path as a string
data_dir = Path("../../data/")

# Convert to a Path object
ratings_path = Path(data_dir / "raw/mal/users-score-2023.csv")
anime_path = Path(data_dir / "raw/mal/anime-dataset-2023.csv")
output_path = Path(data_dir / "processed/attentionrec/mal-dataset.pkl")


# Config
rating_threshold = 7


# Load data
ratings_df = pd.read_csv(ratings_path)
anime_df = pd.read_csv(anime_path)


# Filter valid anime
anime_df = anime_df[anime_df["Type"].isin(["TV", "Movie"])]
anime_df = anime_df.sort_values("anime_id")

anime_ids = anime_df['anime_id'].tolist()
assert len(anime_ids) == len(set(anime_ids)), "Duplicate anime_id detected!"

n_items = len(anime_ids)


# Filter ratings to valid anime
ratings_df = ratings_df[ratings_df["anime_id"].isin(anime_ids)]


# Keep only users with at least one positive
users_with_positive = ratings_df[ratings_df["rating"] >= rating_threshold]["user_id"].unique()
ratings_df = ratings_df[ratings_df["user_id"].isin(users_with_positive)]
user_ids = sorted(ratings_df["user_id"].unique())
n_users = len(user_ids)


# Build interaction lists
positive_interactions = []
negative_interactions = []

for _, group in ratings_df.groupby("user_id"):
    pos = group[group["rating"] >= rating_threshold]["anime_id"].tolist()
    neg = group[group["rating"] < rating_threshold]["anime_id"].tolist()

    positive_interactions.append(pos)
    negative_interactions.append(neg)


# Build all_items list
all_items = list(range(n_items))


# Save everything
dataset = {
    "pos_interactions": positive_interactions,
    "neg_interactions": negative_interactions,
    "anime_ids": anime_ids,  # Store original anime_ids for sampling
}

with open(output_path, "wb") as f:
    pickle.dump(dataset, f)

print(f"Saved processed dataset to {output_path}")
print(f"n_users = {n_users}, n_items = {n_items}")