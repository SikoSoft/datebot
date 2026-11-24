from __future__ import annotations

import sqlite3
from functools import lru_cache

import numpy as np
from PIL import Image

from datebot import db

# Runs fully locally; weights are downloaded once from the Hugging Face hub, photos never leave the machine.
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
EMBED_ID = f"{MODEL_NAME}/{PRETRAINED}"


@lru_cache(maxsize=1)
def _load():
    import open_clip
    import torch

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
    return model.eval().to(device), preprocess, device


def embed_images(paths: list[str], batch_size: int = 16) -> np.ndarray:
    """Return L2-normalized CLIP embeddings, shape (len(paths), dim)."""
    import torch

    model, preprocess, device = _load()
    out = []
    for i in range(0, len(paths), batch_size):
        batch = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in paths[i : i + batch_size]])
        with torch.no_grad():
            feats = model.encode_image(batch.to(device))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        out.append(feats.cpu().numpy().astype(np.float32))
    return np.concatenate(out) if out else np.empty((0, 0), dtype=np.float32)


def embed_pending(conn: sqlite3.Connection) -> int:
    """Embed every photo that has no embedding for the current model. Returns the count embedded."""
    pending = db.photos_missing_embeddings(conn, EMBED_ID)
    if not pending:
        return 0
    vecs = embed_images([p for _, p in pending])
    for (photo_id, _), vec in zip(pending, vecs):
        db.set_embedding(conn, photo_id, vec, EMBED_ID)
    return len(pending)
