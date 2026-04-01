from .recvae import RecVAERetriever


def build_baseline(name, **kwargs):
    if name == "recvae":
        return RecVAERetriever(**kwargs)

    raise ValueError(f"Unknown baseline: {name}")