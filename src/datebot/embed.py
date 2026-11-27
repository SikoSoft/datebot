from __future__ import annotations

import sqlite3
from functools import lru_cache

import numpy as np
from PIL import Image

from datebot import db

# Runs fully locally; weights are downloaded once from the Hugging Face hub, photos never leave the machine.
MODEL_NAME = "ViT-L-14"  # larger than ViT-B-32: AUC 0.70 -> 0.77 on the same ratings
PRETRAINED = "laion2b_s32b_b82k"
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


def embed_texts(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """L2-normalized CLIP text embeddings, shape (len(texts), dim). Text beyond CLIP's 77 tokens is truncated."""
    import open_clip
    import torch

    model, _, device = _load()
    tokenizer = open_clip.get_tokenizer(MODEL_NAME)
    out = []
    for i in range(0, len(texts), batch_size):
        tokens = tokenizer(texts[i : i + batch_size]).to(device)
        with torch.no_grad():
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        out.append(feats.cpu().numpy().astype(np.float32))
    return np.concatenate(out) if out else np.empty((0, 0), dtype=np.float32)


def pool_text(vecs: np.ndarray) -> np.ndarray:
    """Pool one profile's per-line text embeddings (n_lines, dim) into [mean, max] (2 * dim).

    Mean is the overall tone; max keeps one standout prompt from being averaged away."""
    return np.concatenate([vecs.mean(axis=0), vecs.max(axis=0)]).astype(np.float32)


def embed_pending(conn: sqlite3.Connection) -> int:
    """Embed every photo and profile text that lacks an embedding for the current model.

    Returns the number of photos embedded (texts are embedded too, but not counted)."""
    texts = db.profiles_missing_text_embeddings(conn, EMBED_ID)
    if texts:
        lines = embed_texts([line for _, parts in texts for line in parts])  # every line of every profile, one batch
        start = 0
        for pid, parts in texts:
            db.set_text_embedding(conn, pid, pool_text(lines[start : start + len(parts)]), EMBED_ID)
            start += len(parts)
        conn.commit()  # release the write lock before the slow photo embedding below
    pending = db.photos_missing_embeddings(conn, EMBED_ID)
    if not pending:
        return 0
    vecs = embed_images([p for _, p in pending])
    for (photo_id, _), vec in zip(pending, vecs):
        db.set_embedding(conn, photo_id, vec, EMBED_ID)
    conn.commit()
    return len(pending)
