import torch.nn as nn
import torch
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout):
        super().__init__()
        # not sure why we have layernorm before attention but seems to be common in some implementations (pre-norm)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.mha = nn.MultiheadAttention(
            embedding_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.ln2 = nn.LayerNorm(embedding_dim)

        # feedforward with expansion and dropout
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.GELU(),  # supposedly better than ReLU for transformers
            nn.Dropout(dropout),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, key_padding_mask=None):
        # Self-attention block
        h = self.ln1(x)
        attn_out, _ = self.mha(h, h, h, key_padding_mask=key_padding_mask)
        x = x + attn_out

        # Feedforward block
        h = self.ln2(x)
        x = x + self.ffn(h)

        return x

# more powerful model
class TransformerRecommendationModel(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        num_layers=2,
        dropout_rate=0.2,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embedding_dim, num_heads, dropout_rate)
            for _ in range(num_layers)
        ])

        # Final normalization
        self.final_ln = nn.LayerNorm(embedding_dim)

        # Attention pooling
        self.attn_pool = nn.Linear(embedding_dim, 1)

        # Output projection
        self.fc_out = nn.Linear(embedding_dim, embedding_dim) # final user representation projection

    # here the mask is True for positions that should be masked (e.g. items without descriptions))
    def forward(self, item_embeddings, key_padding_mask=None):
        """
        item_embeddings: [batch, num_items, dim]
        """

        x = item_embeddings

        # Transformer stack
        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask)

        x = self.final_ln(x)

        # Pooling
        scores = self.attn_pool(x)  # [B, N, 1]
        weights = F.softmax(scores / 0.7, dim=1)  # temperature helps?
        user_rep = (x * weights).sum(dim=1)

        # Final projection + normalization
        user_rep = F.normalize(self.fc_out(user_rep), p=2, dim=-1)

        return user_rep
    
    def score(self, user_rep, item_embeddings):
        if item_embeddings.dim() == 2:
            return torch.sum(user_rep * item_embeddings, dim=-1)
        elif item_embeddings.dim() == 3:
            return torch.bmm(item_embeddings, user_rep.unsqueeze(-1)).squeeze(-1)

# Note: the evaluation code is designed to work with any model that has a .score() method which computes dot product scores between user representations and item embeddings. 
# So we can directly use this new model without changing the evaluation logic.    
class AttentionRecommendationModel(nn.Module):
    def __init__(
        self,
        embedding_dim,
        num_heads,
        dropout_rate=0.2,
        pooling='attention',  # 'attention' or 'mean'/'sum'/'max'
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.pooling = pooling

        # Multi-head attention over items
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True
        )

        # Optional feature-level weighting
        if self.use_feature_weighting:
            self.feature_fc1 = nn.Linear(embedding_dim, 4 * embedding_dim)
            self.feature_fc2 = nn.Linear(4 * embedding_dim, embedding_dim)

        # Feedforward for pooled user embedding
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, 4 * embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(4 * embedding_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

        # Final output transformation
        self.fc_out = nn.Linear(embedding_dim, embedding_dim)

        # Attention pooling weights
        if self.pooling == 'attention':
            self.attn_pool = nn.Linear(embedding_dim, 1)

    # TODO: add key_padding_mask to handle items without descriptions (mask out in attention and pooling)
    def forward(self, item_embeddings):
        """
        item_embeddings: [batch_size, num_items, embedding_dim]
        """
        # Multi-head self-attention
        attn_output, _ = self.multihead_attention(
            item_embeddings, item_embeddings, item_embeddings
        )  # [batch, num_items, embedding_dim]

        # Pooling across items
        if self.pooling == 'mean':
            user_rep = attn_output.mean(dim=1)
        elif self.pooling == 'max':
            user_rep, _ = attn_output.max(dim=1)
        elif self.pooling == 'sum':
            user_rep = attn_output.sum(dim=1)
        elif self.pooling == 'attention':
            scores = self.attn_pool(attn_output)  # [batch, num_items, 1]
            weights = F.softmax(scores, dim=1)
            user_rep = (attn_output * weights).sum(dim=1)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling}")

        # Feedforward transformation
        user_rep = self.feedforward(user_rep)

        # Final linear + L2 normalization
        user_rep = F.normalize(self.fc_out(user_rep), p=2, dim=-1)

        return user_rep

    def score(self, user_rep, item_embeddings):
        """
        Compute dot product scores between user and item embeddings.

        user_rep: [batch_size, embedding_dim]
        item_embeddings: [batch_size, num_items, embedding_dim] or [batch_size, embedding_dim]

        Returns:
        - [batch_size] or [batch_size, num_items]
        """
        if item_embeddings.dim() == 2:  # single positive item
            return torch.sum(user_rep * item_embeddings, dim=-1)
        elif item_embeddings.dim() == 3:  # multiple negatives
            return torch.bmm(item_embeddings, user_rep.unsqueeze(-1)).squeeze(-1)
