import numpy as np
import random 
import torch

def hit_rate(rank, K=10):
    return 1.0 if rank <= K else 0.0

def ndcg(rank, K=10):
    if rank <= K:
        return 1 / np.log2(rank + 1)
    return 0.0

def evaluate_ranks(ranks, K=10):
    hr = np.mean([hit_rate(r, K) for r in ranks])
    ndcg_score = np.mean([ndcg(r, K) for r in ranks])
    return {
        "HR@{}".format(K): hr,
        "NDCG@{}".format(K): ndcg_score
    }