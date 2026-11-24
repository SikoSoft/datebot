import numpy as np
import pytest

from datebot import db


@pytest.fixture
def conn(tmp_path):
    c = db.connect(str(tmp_path / "t.db"))
    yield c
    c.close()


def test_add_profile_dedupes_on_external_id(conn):
    a = db.add_profile(conn, "tinder", ["a.jpg"], external_id="x1", name="A")
    b = db.add_profile(conn, "tinder", ["b.jpg"], external_id="x1", name="B")
    assert a == b
    assert db.photo_paths(conn, a) == ["a.jpg"]


def test_latest_rating_wins(conn):
    pid = db.add_profile(conn, "manual", ["a.jpg"])
    db.add_rating(conn, pid, False)
    db.add_rating(conn, pid, True)
    assert db.latest_ratings(conn) == {pid: True}


def test_unrated_and_clear(conn):
    p1 = db.add_profile(conn, "manual", ["a.jpg"])
    p2 = db.add_profile(conn, "manual", ["b.jpg"])
    db.add_rating(conn, p1, True)
    assert db.unrated_profile_ids(conn) == [p2]
    assert db.clear_ratings(conn) == 1
    assert db.unrated_profile_ids(conn) == [p1, p2]


def test_embedding_roundtrip_and_model_mismatch(conn):
    pid = db.add_profile(conn, "manual", ["a.jpg", "b.jpg"])
    ids = [i for i, _ in db.photos_missing_embeddings(conn, "m1")]
    for n, photo_id in enumerate(ids):
        db.set_embedding(conn, photo_id, np.full(4, n, dtype=np.float32), "m1")
    assert db.photos_missing_embeddings(conn, "m1") == []
    assert len(db.photos_missing_embeddings(conn, "m2")) == 2
    emb = db.profile_embeddings(conn, pid, "m1")
    assert emb.shape == (2, 4)
    assert db.profile_embeddings(conn, pid, "m2").size == 0
