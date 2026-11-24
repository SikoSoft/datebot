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
    assert vec.shape == (512,) and abs(np.linalg.norm(vec) - 1) < 1e-3
    assert db.text_embedding(conn, silent, embed.EMBED_ID) is None
    assert db.profiles_missing_text_embeddings(conn, embed.EMBED_ID) == []
