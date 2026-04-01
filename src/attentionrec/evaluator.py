import torch
from tqdm import tqdm
from utils.metrics import evaluate_ranks

# TODO: we need fix the full catalog metrics.
class Evaluator:
    def __init__(
        self,
        description_embeddings,
        anime_id_to_embedding_idx,
        embedding_idx_to_anime_id,
        device,
        baseline_model=None,
        candidate_K=None,
    ):
        self.description_embeddings = description_embeddings
        self.anime_id_to_embedding_idx = anime_id_to_embedding_idx
        self.embedding_idx_to_anime_id = embedding_idx_to_anime_id
        self.device = device
        self.baseline_model = baseline_model
        self.candidate_K = candidate_K
        self._candidate_cache: dict[int, list[int]] = {}  # user_id → embedding indices

    
    # Precompute candidates
    def precompute_candidates(self, eval_loader):
        """
        Populates _candidate_cache: {user_id: [embedding_idx, ...]}
        Keyed by user_id so evaluation is order-independent.
        """
        assert self.baseline_model is not None, "baseline_model is None!"
        assert self.candidate_K is not None, "candidate_K not set!"

        self._candidate_cache.clear()

        for batch in tqdm(eval_loader, desc="Precomputing candidates"):
            user_ids, user_items_tensor, _ = batch

            user_anime_ids = [
                [self.embedding_idx_to_anime_id[idx.item()] for idx in items]
                for items in user_items_tensor
            ]
            candidate_lists = self.baseline_model.recommend_batch(
                user_anime_ids, K=self.candidate_K
            )
            
            for user_id, candidates in zip(user_ids, candidate_lists):
                self._candidate_cache[user_id.item()] = [
                    self.anime_id_to_embedding_idx[it]
                    for it in candidates
                    if it in self.anime_id_to_embedding_idx
                ]

    def _get_candidates(self, user_ids: list[int]) -> list[list[int]]:
        """Fetch cached candidates for a batch; raises clearly if missing."""
        missing = [uid for uid in user_ids if uid not in self._candidate_cache]
        if missing:
            raise RuntimeError(
                f"No cached candidates for user_ids {missing[:5]}{'...' if len(missing) > 5 else ''}. "
                "Call precompute_candidates() first."
            )
        return [self._candidate_cache[uid] for uid in user_ids]

    
    # Evaluate
    def evaluate(self, model, eval_loader, eval_Ks=[10, 25, 50]):
        model.eval()
        ranks = []
        total = recall_hits = 0

        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating"):
                user_ids, user_items_tensor, test_items_tensor = batch
                batch_size = len(user_items_tensor)
                user_ids_list = [uid.item() for uid in user_ids]

                test_items_tensor = test_items_tensor.to(self.device)
                user_items_tensor = [items.to(self.device) for items in user_items_tensor]

                # Build user representations
                user_reps, mask_indices = [], []
                for items in user_items_tensor:
                    user_rep = model(self.description_embeddings[items].unsqueeze(0)).squeeze(0)
                    user_reps.append(user_rep)
                    mask_indices.append(items.tolist())
                user_reps_tensor = torch.stack(user_reps)  # [B, D]

                if self.baseline_model is not None:
                    batch_ranks, batch_total, batch_hits = self._evaluate_retriever_path(
                        user_ids_list, user_reps_tensor, mask_indices, test_items_tensor, batch_size
                    )
                    ranks.extend(batch_ranks)
                    total += batch_total
                    recall_hits += batch_hits
                else:
                    batch_ranks = self._evaluate_full_catalog_path(
                        user_reps_tensor, mask_indices, test_items_tensor, batch_size
                    )
                    ranks.extend(batch_ranks)

        return self._compute_metrics(ranks, eval_Ks, total, recall_hits)

    
    # Retriever path
    def _evaluate_retriever_path(
        self, user_ids, user_reps_tensor, mask_indices, test_items_tensor, batch_size
    ):
        candidate_idx_lists = self._get_candidates(user_ids)
        
        max_len = max((len(lst) for lst in candidate_idx_lists), default=1)
        D = self.description_embeddings.shape[1]

        cand_embs = torch.zeros(batch_size, max_len, D, device=self.device)
        cand_mask = torch.zeros(batch_size, max_len, dtype=torch.bool, device=self.device)

        for i, idx_list in enumerate(candidate_idx_lists):
            if idx_list:
                idx_tensor = torch.tensor(idx_list, device=self.device)
                cand_embs[i, : len(idx_list)] = self.description_embeddings[idx_tensor]
                cand_mask[i, : len(idx_list)] = True

        # Score all candidates in one bmm
        scores = torch.bmm(cand_embs, user_reps_tensor.unsqueeze(2)).squeeze(2)
        scores[~cand_mask] = -torch.inf

        # Mask seen items
        for i, (seen, idx_list) in enumerate(zip(mask_indices, candidate_idx_lists)):
            pos_of = {idx: pos for pos, idx in enumerate(idx_list)}
            for idx in seen:
                if idx in pos_of:
                    scores[i, pos_of[idx]] = -torch.inf

        # Ground-truth scores
        test_embs = self.description_embeddings[test_items_tensor]
        gt_scores = (user_reps_tensor * test_embs).sum(dim=1)

        ranks, total, hits = [], 0, 0
        for i in range(batch_size):
            gt_idx = test_items_tensor[i].item()
            total += 1
            if gt_idx in candidate_idx_lists[i]:
                hits += 1
                rank = 1 + (scores[i] > gt_scores[i]).sum().item()
            else:
                rank = len(candidate_idx_lists[i]) + 1
            ranks.append(rank)

        return ranks, total, hits

    
    # Full-catalog path
    def _evaluate_full_catalog_path(
        self, user_reps_tensor, mask_indices, test_items_tensor, batch_size
    ):
        scores = user_reps_tensor @ self.description_embeddings.T

        # Sanity checks
        for i in range(batch_size):
            gt = test_items_tensor[i].item()
            assert gt not in mask_indices[i], f"Test item {gt} is in mask_indices for user {i}!"
            assert gt < scores.shape[1], f"Test item {gt} out of range (catalog size {scores.shape[1]})"
        
        gt_scores = scores[torch.arange(batch_size), test_items_tensor]

        for i, seen in enumerate(mask_indices):
            scores[i, seen] = -torch.inf

        ranks = ((scores > gt_scores[:, None]).sum(dim=1) + 1).cpu().tolist()
        return ranks
    
    
    # Metrics
    def _compute_metrics(self, ranks, eval_Ks, total, recall_hits):
        metrics = {}
        for K in eval_Ks:
            for key, val in evaluate_ranks(ranks, K=K).items():
                metrics[f"rerank_{key}"] = val

        if self.baseline_model is not None and total > 0:
            recall = recall_hits / total
            metrics[f"recall@{self.candidate_K}"] = recall
            for K in eval_Ks:
                for prefix in ("NDCG", "HR", "MRR"):
                    rerank_key = f"rerank_{prefix}@{K}"
                    if rerank_key in metrics:
                        metrics[f"final_{prefix}@{K}"] = recall * metrics[rerank_key]

        return metrics