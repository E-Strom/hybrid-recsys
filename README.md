# Transformer-Based Anime Recommender System

A deep learning recommender system that leverages transformer-based user modeling and contrastive learning (InfoNCE) to rerank candidate items based on user interaction history.

This project implements an optional two-stage recommendation pipeline:
1. **Candidate Retrieval** (e.g., using a baseline like RecVAE or Matrix Factorization)
2. **Transformer-Based Reranking** (fine-grained scoring with self-attention)

Here, we explore a hybrid between:

- Collaborative filtering (user history)
- Content-based recommendation (text embeddings)
- Prompt-based control (natural language steering)

It demonstrates how modern embedding models enable a unified
representation space where users, items, and queries can interact.


## Overview

Traditional recommender systems rely on static user embeddings that fail to capture the dynamic, contextual relationships between the items a user has watched. 


This project models users as sets of interacted items and learns a dynamic user representation using a Transformer. The model is trained with a contrastive objective, pulling the generated user representation closer to positive items and pushing it away from negatives in the shared embedding space.

Beyond standard history-based recommendations, since the item embeddings are derived from a pre-trained sentence transformer (stsb-roberta-large), we can inject arbitrary text prompts directly into the user representation.

### How it works:
1. A user’s history of anime is pulled and converted into a sequence of precomputed description embeddings.
2. A natural language prompt (e.g., *"Magical girl anime"*) is passed through the same transformer.
3. The prompt embedding lives in the same semantic space and is appended to the sequence of item embeddings.
4. The Transformer jointly attends over both historical items and the prompt, producing a context-aware user representation conditioned on the query.

A natural extension is to incorporate multimodal embeddings (e.g., CLIP), enabling the model to capture visual style and aesthetic features alongside textual semantics.

## Model Architecture

### TransformerRecommendationModel
* **Input:** Item embeddings from user history with shape `[batch_size, B, embedding_dim]`.
* Stacked Transformer blocks featuring pre-layer normalization Multi-head Self-Attention and GELU-activated feedforward networks.
* Attention-based pooling with a custom temperature-scaled softmax to create a unified user representation from the attended sequence.
* Final projection and $L_2$ normalization.

## Training Objective

### Contrastive Loss (InfoNCE)
We utilize a temperature-scaled contrastive loss to pull the user vector closer to positive items and push it away from batch-sampled negatives:

$$L = -\log\left( \frac{\exp(\text{sim}(u, pos)/\tau)}{\sum_{i} \exp(\text{sim}(u, i)/\tau)} \right)$$

Where:
* $u$ is the user representation.
* $pos$ is the positive item embedding.
* $i$ iterates over both the positive and all negative sampled items.
* $\tau$ is the temperature hyperparameter (default: 0.07).

## Notebooks and Inference

The repository includes detailed Jupyter notebooks modeling the entire pipeline:
* **`recVAE_example.ipynb`**: Demonstrates the baseline variational autoencoder used for candidate retrieval.
* **`attentionrec_example.ipynb`**: Demonstrates loading the trained Transformer, generating synthetic user archetypes, and performing prompt-based augmented user embedding inference.

## Data & Current Limitations

While the architecture is highly flexible, it is important to note the present constraints of the data:
* **Synopsis-Driven Embeddings:** Recommendations and item representations are derived strictly from text input.
* **The "MAL Synopsis" Bottleneck:** The dataset relies on MyAnimeList synopses. A brief synopsis is rarely the most holistic representation of an anime's genre, tone, or artistic direction. Because of this, pure catalog recommendations may sometimes feel slightly noisy, as the model relies on overlapping keyword themes rather than deep stylistic metadata.
* **The Solution:** The architecture is entirely data-agnostic. The pipeline is fully prepared to accept richer textual data (such as aggregated user reviews) or multi-modal embeddings (visual poster vectors) without altering the model's core code.

## Future Work

- Multimodal embeddings (e.g., CLIP for poster + text fusion)
- Cross-attention between prompts and user history
- End-to-end fine-tuning of embedding model
- Diffusion / generative recommenders (emerging direction)

## Acknowledgements

The core architecture for modeling users as a set of attended items in a contrastive setting was inspired by Section 3.3 of the paper: *"Transformer-Empowered Content-Aware Collaborative Filtering"* [arXiv:2204.00849](https://arxiv.org/abs/2204.00849).