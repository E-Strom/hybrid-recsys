import torch
from .base import BaseRetriever
import sys
import numpy as np

sys.path.append("../../RecVAE")  # path to RecVAE repo
from model import VAE as RecVAE

class RecVAERetriever(BaseRetriever):
    def __init__(self, checkpoint_path, device="cuda"):
        self.device = torch.device(device)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.itemid2idx = checkpoint['itemid2idx']
        self.idx2itemid = {v: k for k, v in self.itemid2idx.items()} # we need to project back to animeID space before returning list
        self.num_items = len(self.itemid2idx)       # baseline-specific mapping

        # Load model
        self.model = self._load_model(checkpoint)
        self.model.eval()

    def _load_model(self, checkpoint):
        model = RecVAE(**checkpoint['model_kwargs'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        return model

    def _build_user_vector(self, user_items):
        """
        Build multi-hot user vector.
        Args:
            user_items: list of raw item IDs
        Returns:
            vec: multi-hot tensor of shape [num_items]
            indices: list of indices corresponding to seen items
        """
        indices = [self.itemid2idx[iid] for iid in user_items if iid in self.itemid2idx]
        vec = torch.zeros(self.num_items, device=self.device)
        if indices:
            vec[indices] = 1.0
        return vec, indices

    def recommend(self, user_items: list[int], K: int):
        """
        Single-user top-K recommendation
        """
        user_vec, seen_indices = self._build_user_vector(user_items)
        with torch.no_grad():
            scores = self.model(user_vec.unsqueeze(0), calculate_loss=False).squeeze(0)
        if seen_indices:
            scores[seen_indices] = -float("inf")
        topk_indices = torch.topk(scores, K).indices.tolist()
        return [self.idx2itemid[idx] for idx in topk_indices]

    def recommend_batch(self, batch_user_items: list[list[int]], K: int):
        """
        Batch recommendation for multiple users
        """
        batch_size = len(batch_user_items)
        user_vectors = torch.zeros(batch_size, self.num_items, device=self.device)
        mask_indices = []

        for i, user_items in enumerate(batch_user_items):
            vec, seen = self._build_user_vector(user_items)
            user_vectors[i] = vec
            mask_indices.append(seen)

        with torch.no_grad():
            scores = self.model(user_vectors, calculate_loss=False)  # [batch_size, num_items]

        # Mask already-seen items per user
        for i, seen in enumerate(mask_indices):
            if seen:
                scores[i, seen] = -float("inf")

        # Get top-K indices per user
        topk_indices = torch.topk(scores, K, dim=1).indices  # [batch_size, K]
        return [
            [self.idx2itemid[idx] for idx in topk_indices[i].tolist()]
            for i in range(batch_size)
        ]