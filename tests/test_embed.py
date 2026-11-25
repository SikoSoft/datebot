import numpy as np
import pytest
from PIL import Image

pytest.importorskip("open_clip")

from datebot import db, embed  # noqa: E402


@pytest.mark.slow
def test_embed_pending_stores_normalized_vectors(tmp_path):
    img = tmp_path / "a.png"
    Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8")).save(img)
    conn = db.connect(str(tmp_path / "t.db"))
    pid = db.add_profile(conn, "manual", [str(img)])
    assert embed.embed_pending(conn) == 1
    assert embed.embed_pending(conn) == 0  # idempotent
    emb = db.profile_embeddings(conn, pid, embed.EMBED_ID)
    assert emb.shape == (1, 512)
    assert abs(np.linalg.norm(emb[0]) - 1) < 1e-3


@pytest.mark.slow
def test_embed_pending_embeds_bio_text(tmp_path):
    img = tmp_path / "a.png"
    Image.fromarray((np.random.rand(64, 64, 3) * 255).astype("uint8")).save(img)
    conn = db.connect(str(tmp_path / "t.db"))
    hiker = db.add_profile(conn, "manual", [str(img)], bio="I love hiking and camping")
    silent = db.add_profile(conn, "manual", [str(img)], external_id="b")
    embed.embed_pending(conn)
    vec = db.text_embedding(conn, hiker, embed.EMBED_ID)
    assert vec.shape == (1024,)  # [mean, max] of the per-line embeddings
    assert db.text_embedding(conn, silent, embed.EMBED_ID) is None
    assert db.profiles_missing_text_embeddings(conn, embed.EMBED_ID) == []


def test_pool_text_is_mean_then_max():
    pooled = embed.pool_text(np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float32))
    assert list(pooled) == [0.5, 1.0, 1.0, 2.0]


def test_each_text_line_is_embedded_separately_and_old_layouts_are_redone(tmp_path, monkeypatch):
    seen = []

    def fake_texts(texts):
        seen.append(list(texts))
        return np.eye(4, dtype=np.float32)[: len(texts)]

    monkeypatch.setattr(embed, "embed_texts", fake_texts)
    conn = db.connect(str(tmp_path / "t.db"))
    pid = db.add_profile(conn, "hinge", [], bio="Hi", extra={"prompts": [{"q": "Q", "a": "tea"}]})
    embed.embed_pending(conn)
    assert seen == [["Hi", "Q", "tea"]]  # bio and every prompt line on its own, so none is truncated away
    assert db.text_embedding(conn, pid, embed.EMBED_ID).shape == (8,)
    # a vector stored under an older layout is stale and gets re-embedded from the stored text
    conn.execute("UPDATE profiles SET text_embed_model = ? WHERE id = ?", (embed.EMBED_ID, pid))
    assert db.text_embedding(conn, pid, embed.EMBED_ID) is None
    embed.embed_pending(conn)
    assert len(seen) == 2 and db.text_embedding(conn, pid, embed.EMBED_ID) is not None
