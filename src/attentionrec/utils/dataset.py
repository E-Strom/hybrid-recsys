import numpy as np
import torch
from torch.utils.data import Dataset


class AnimeDataset(Dataset):
    """
    here we assume we get anime IDs and we then return indices in embedding space
    """
    def __init__(self, pos_interactions, neg_interactions, anime_id_to_idx, num_catalog_items, N, B, neutral_ratio=1.0):
        self.num_catalog_items = num_catalog_items
        
        # map to embedding space
        self.pos_interactions = [
            [anime_id_to_idx[a] for a in user if a in anime_id_to_idx] for user in pos_interactions
        ]
        self.neg_interactions = [
            [anime_id_to_idx[a] for a in user if a in anime_id_to_idx] for user in neg_interactions
        ]
        
        self.n_users = len(pos_interactions)
        self.N = N
        self.B = B
        self.neutral_ratio = neutral_ratio

        all_items = set(range(num_catalog_items))  # {0, 1, ..., num_catalog_items - 1}

        self.user_interactions = []
        self.true_neutral = []

        for u in range(self.n_users):
            pos = set(self.pos_interactions[u])
            neg = set(self.neg_interactions[u])
            interacted = pos | neg
            self.user_interactions.append(list(interacted))
            self.true_neutral.append(list(all_items - interacted))


    def __len__(self):
        return self.n_users

    def __getitem__(self, user_idx):
        pos_interactions = self.pos_interactions[user_idx]
        neg_interactions = self.neg_interactions[user_idx]
        neutral_pool = self.true_neutral[user_idx]

        # this will never happen
        assert len(neutral_pool) > 0, f"User {user_idx} has interacted with all catalog items"

        user_interactions = self.user_interactions[user_idx]

        # User representation
        replace = len(user_interactions) < self.B
        user_samples = np.random.choice(user_interactions, self.B, replace=replace)

        # Positive sample
        pos_item = np.random.choice(pos_interactions)

        # Negative sampling
        neg_samples = []

        max_explicit_neg = int(self.N * (1 - self.neutral_ratio))
        if neg_interactions:
            explicit_count = min(len(neg_interactions), max_explicit_neg)
            if explicit_count > 0:
                neg_samples.extend(np.random.choice(neg_interactions, explicit_count, replace=False))

        remaining = self.N - len(neg_samples)
        if remaining > 0:
            replace = remaining > len(neutral_pool)
            neg_samples.extend(np.random.choice(neutral_pool, remaining, replace=replace))

        # Convert to tensors
        return (
            user_idx,
            torch.tensor(user_samples, dtype=torch.long),
            torch.tensor(pos_item, dtype=torch.long),
            torch.tensor(neg_samples, dtype=torch.long),
        )

class AnimeEvalDataset(Dataset):
    def __init__(self, train_interactions, test_items, anime_id_to_idx):
        """
        here we assume we get anime IDs and we then return indices in embedding space

        train_interactions: list[list[idx]]
        test_items: list[idx] 
        """
        self.train_interactions = train_interactions
        self.test_items = test_items

        # map to embedding space
        self.train_interactions = [
            [anime_id_to_idx[i] for i in user if i in anime_id_to_idx] for user in train_interactions
        ]
        self.test_items = [anime_id_to_idx[i] for i in test_items if i in anime_id_to_idx]

        self.n_users = len(train_interactions)

    def __len__(self):
        return self.n_users

    def __getitem__(self, user_idx):
        user_items = self.train_interactions[user_idx]
        test_item = self.test_items[user_idx]

        user_items_tensor = torch.tensor(user_items, dtype=torch.long)
        test_item_tensor = torch.tensor([test_item], dtype=torch.long)  # always 1-element tensor

        return user_idx, user_items_tensor, test_item_tensor

# Leave-one-out split
def leave_one_out_split(pos_interactions):
    
    # embedding space
    train_interactions = []
    test_items = []

    for items in pos_interactions:
        if len(items) < 2:
            continue  # skip users without enough interactions

        items = list(items)
        test_item = items[-1]  # deterministic last item
        train_items = items[:-1]

        train_interactions.append(train_items)
        test_items.append(test_item)

    return train_interactions, test_items

# Collate function for eval
def eval_collate_fn(batch):
    user_ids, user_items, test_items = zip(*batch)
    user_ids = torch.tensor(user_ids)
    test_items = torch.cat(test_items, dim=0)  # flatten 1-element tensors
    user_items = list(user_items)  # keep as list of tensors

    return user_ids, user_items, test_items