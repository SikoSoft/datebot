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
