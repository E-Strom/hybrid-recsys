
import argparse
import json
import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
import multiprocessing
from datetime import datetime
from argparse import Namespace

from baselines.factory import build_baseline
from utils.dataset import AnimeDataset
from utils.dataset import AnimeEvalDataset
from utils.dataset import leave_one_out_split, eval_collate_fn

from models.attentionrec import TransformerRecommendationModel
from losses.contrastive_loss import ContrastiveLoss
from evaluator import Evaluator

def parse_args():
    parser = argparse.ArgumentParser(description="Anime Recommender Training")
    
    # Minimal essential CLI arguments
    parser.add_argument("--dataset_path", type=str, required=True, help="Pickled dataset path")
    parser.add_argument("--embeddings_path", type=str, required=True, help="Pickled embeddings path")
    parser.add_argument("--experiment_name", type=str, default="runs", help="Experiment folder name")
    parser.add_argument("--config_path", type=str, default=None,
                        help="Optional JSON config file to override defaults")
    return parser.parse_args()


def load_config(args):
    """Load hyperparameters from JSON config, fallback to defaults if needed"""
    # Default hyperparameters
    config = {
        "baseline": None,
        "baseline_path": None,
        "candidate_K": 100,
        "n_epochs": 100,
        "batch_size": 1024,
        "lr": 1e-3,
        "eval_every": 10, # evel every N epochs
        "B": 8,
        "N": 16,
        "num_heads": 8,
        "dropout_rate": 0.1,
        "pooling": "attention",
        "temperature": 0.07,
        "neutral_ratio": 0.8,
        "eval_Ks": [10, 25, 50]
    }
    
    if args.config_path:
        if not os.path.exists(args.config_path):
            raise FileNotFoundError(f"Config file not found: {args.config_path}")
        with open(args.config_path, "r") as f:
            file_config = json.load(f)
        config.update(file_config)
    
    return config

def train(config):
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Unpack config
    dataset_path = config.dataset_path
    embeddings_path = config.embeddings_path
    experiment_name = config.experiment_name
    baseline = config.baseline
    baseline_path = config.baseline_path
    candidate_K = config.candidate_K
    n_epochs = config.n_epochs
    batch_size = config.batch_size
    lr = config.lr
    eval_every = config.eval_every
    B = config.B
    N = config.N
    num_heads = config.num_heads
    dropout_rate = config.dropout_rate
    pooling = config.pooling
    temperature = config.temperature
    neutral_ratio = config.neutral_ratio
    eval_Ks = config.eval_Ks

    if baseline_path is not None:
        assert candidate_K is not None, "candidate_K must be set if using baseline_model"

    # TensorBoard folder with timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"runs_pytorch/{experiment_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=save_dir)

    # Save config
    config_path = os.path.join(save_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(vars(config), f, indent=2)

    # Load embeddings
    with open(embeddings_path, "rb") as f:
        emb_data = pickle.load(f)

    description_embeddings = torch.tensor(emb_data['embeddings']).to(device)
    embedding_dim = emb_data['embedding_dim']
    anime_id_to_idx = emb_data['anime_id_to_idx']
    embedding_idx_to_anime_id = {v: k for k, v in anime_id_to_idx.items()}

    # Load dataset
    with open(dataset_path, "rb") as f:
        dataset_data = pickle.load(f)

    pos_interactions = dataset_data["pos_interactions"]
    neg_interactions = dataset_data["neg_interactions"]


    # Leave-one-out split
    train_interactions, test_items = leave_one_out_split(pos_interactions=pos_interactions)

    
    # Dataset & DataLoaders
    num_workers = min(multiprocessing.cpu_count(), batch_size)

    train_dataset = AnimeDataset(
        pos_interactions=train_interactions,
        neg_interactions=neg_interactions,
        anime_id_to_idx=anime_id_to_idx, # embedding space projection
        num_catalog_items=len(description_embeddings),
        N=N,
        B=B,
        neutral_ratio=neutral_ratio
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        drop_last=True, num_workers=num_workers
    )

    eval_dataset = AnimeEvalDataset(
        train_interactions=train_interactions,
        test_items=test_items,
        anime_id_to_idx=anime_id_to_idx
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=eval_collate_fn
    )

    
    # Baseline and Evaluator
    baseline_model = None
    if baseline:
        baseline_model = build_baseline(
            baseline, checkpoint_path=baseline_path, device=device
        )

    evaluator = Evaluator(
        description_embeddings=description_embeddings,
        anime_id_to_embedding_idx=anime_id_to_idx,
        embedding_idx_to_anime_id=embedding_idx_to_anime_id,
        device=device,
        baseline_model=baseline_model,
        candidate_K=candidate_K
    )

    if baseline_model is not None:
        print("Precomputing candidates for all eval users...")
        evaluator.precompute_candidates(eval_loader)

    
    # Model, Opt, Scheduler, Loss
    model = TransformerRecommendationModel(
        embedding_dim=embedding_dim,
        num_heads=num_heads,
        num_layers=2, # TODO: config
        dropout_rate=dropout_rate,
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * n_epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
    )
    contrastive_loss = ContrastiveLoss(temperature=temperature)

    best_hr = 0
    
    # Training loop
    for epoch in range(0, n_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")

        for _, user_item_idx, pos_idx, neg_idx in pbar:
            user_item_embeddings = description_embeddings[user_item_idx]
            pos_embeddings = description_embeddings[pos_idx]
            neg_embeddings = description_embeddings[neg_idx]

            optimizer.zero_grad()
            loss = contrastive_loss(model, user_item_embeddings, pos_embeddings, neg_embeddings)
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Average Train Loss = {avg_loss:.4f}")
        writer.add_scalar("Loss/Train", avg_loss, epoch)

        
        # Evaluation
        if (epoch + 1) % eval_every == 0:
            metrics = evaluator.evaluate(
                model=model,
                eval_loader=eval_loader,
                eval_Ks=eval_Ks
            )

            # Primary metrics for console logging
            primary_metrics = ["rerank_HR@10", f"recall@{candidate_K}"]
            print(f"\nEpoch {epoch+1}/{n_epochs} evaluation:")
            for k in primary_metrics:
                v = metrics.get(k, 0.0)
                print(f"  {k}: {v:.8f}")
                writer.add_scalar(f"Metrics/{k}", v, epoch)

            # Log all metrics to TensorBoard
            for k, v in metrics.items():
                writer.add_scalar(f"Metrics/All/{k}", v, epoch)

            # Save best model by HR@10
            hr10 = metrics.get("rerank_HR@10", 0.0)
            if hr10 > best_hr:
                best_hr = hr10
                best_model_path = os.path.join(save_dir, "best_model.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'imizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_hr': best_hr
                }, best_model_path)
                print(f"New best HR@10: {best_hr:.8f} saved to {best_model_path}")

            # Always save latest checkpoint
            latest_checkpoint_path = os.path.join(save_dir, "latest_checkpoint.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_hr': best_hr
            }, latest_checkpoint_path)

    writer.close()

if __name__ == "__main__":
    args = parse_args()
    config_dict = load_config(args)

    # CLI args override JSON config, skip None values so JSON defaults are preserved
    for key, value in vars(args).items():
        if value is not None:
            config_dict[key] = value

    config = Namespace(**config_dict)
    train(config)