import torch
import torch.nn as nn

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        """
        Contrastive loss (InfoNCE) with optional temperature scaling.
        Args:
            temperature (float): scaling factor for logits; typical range 0.05–0.2
        """
        super().__init__()
        self.temperature = temperature

    def forward(self, model, user_item_embeddings, pos_embeddings, neg_embeddings):
        """
        Args:
            model: user/item embedding model
            user_item_embeddings: [batch_size, B, embedding_dim] 
            pos_embeddings: [batch_size, embedding_dim] or [batch_size, 1, embedding_dim]
            neg_embeddings: [batch_size, N, embedding_dim]
        """
        # Compute user representations
        user_rep = model(user_item_embeddings)  # [batch_size, embedding_dim]
    
        # Compute positive and negative scores
        pos_scores = model.score(user_rep, pos_embeddings)  # [batch_size]
        neg_scores = model.score(user_rep, neg_embeddings)  # [batch_size, N]

        # Combine pos + neg for InfoNCE
        all_scores = torch.cat([pos_scores.unsqueeze(1), neg_scores], dim=1)  # [batch_size, 1+N]

        # Apply temperature scaling
        all_scores = all_scores / self.temperature

        # InfoNCE / softmax loss
        loss = - pos_scores / self.temperature + torch.logsumexp(all_scores, dim=1)  # [batch_size]

        return loss.mean()