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


def test_undo_last_rating(conn):
    pid = db.add_profile(conn, "manual", ["a.jpg"])
    assert db.undo_last_rating(conn) is None
    db.add_rating(conn, pid, False)
    db.add_rating(conn, pid, True)
    assert db.undo_last_rating(conn) == pid
    assert db.latest_ratings(conn) == {pid: False}


def test_profile_text_joins_bio_and_prompts(conn):
    pid = db.add_profile(conn, "hinge", ["a.jpg"], bio=" Hi ", extra={"prompts": [{"q": "Q", "a": "I like tea"}]})
    assert db.profile_text(db.get_profile(conn, pid)) == "Hi Q I like tea"
    empty = db.add_profile(conn, "manual", ["b.jpg"])
    assert db.profile_text(db.get_profile(conn, empty)) == ""


def test_text_embedding_roundtrip_skips_empty_text(conn):
    with_text = db.add_profile(conn, "manual", ["a.jpg"], bio="loves hiking")
    db.add_profile(conn, "manual", ["b.jpg"])  # no text -> never needs an embedding
    assert db.profiles_missing_text_embeddings(conn, "m1") == [(with_text, "loves hiking")]
    db.set_text_embedding(conn, with_text, np.arange(4, dtype=np.float32), "m1")
    assert db.profiles_missing_text_embeddings(conn, "m1") == []
    assert db.profiles_missing_text_embeddings(conn, "m2") == [(with_text, "loves hiking")]
    assert list(db.text_embedding(conn, with_text, "m1")) == [0, 1, 2, 3]
    assert db.text_embedding(conn, with_text, "m2") is None


def test_migrates_database_from_before_text_columns(tmp_path):
    import sqlite3

    path = str(tmp_path / "old.db")
    old = sqlite3.connect(path)
    old.execute("CREATE TABLE profiles (id INTEGER PRIMARY KEY, source TEXT NOT NULL, external_id TEXT, name TEXT,"
                " age INTEGER, bio TEXT, extra TEXT NOT NULL DEFAULT '{}', created_at TEXT, UNIQUE (source, external_id))")
    old.execute("INSERT INTO profiles (source, bio) VALUES ('manual', 'old row')")
    old.commit()
    old.close()
    conn = db.connect(path)
    assert db.get_profile(conn, 1).bio == "old row"
    assert db.profiles_missing_text_embeddings(conn, "m") == [(1, "old row")]


def test_find_profile_id(conn):
    pid = db.add_profile(conn, "tinder", ["a.jpg"], external_id="abc")
    assert db.find_profile_id(conn, "tinder", "abc") == pid
    assert db.find_profile_id(conn, "hinge", "abc") is None
    assert db.find_profile_id(conn, "tinder", "zzz") is None


def test_profile_text_includes_interests_and_descriptors(conn):
    pid = db.add_profile(conn, "tinder", ["a.jpg"], bio="Hi",
                         extra={"interests": ["Hiking", "Coffee"], "descriptors": ["Smoking: Non-smoker"],
                                "jobs": ["ignored"]})
    assert db.profile_text(db.get_profile(conn, pid)) == "Hi Hiking Coffee Smoking: Non-smoker"


def test_latest_ratings_are_in_chronological_order(conn):
    a = db.add_profile(conn, "manual", ["a.jpg"])
    b = db.add_profile(conn, "manual", ["b.jpg"])
    c = db.add_profile(conn, "manual", ["c.jpg"])
    for pid in (c, a, b):
        db.add_rating(conn, pid, True)
    db.add_rating(conn, c, False)  # re-rating moves c to the end
    assert list(db.latest_ratings(conn)) == [a, b, c]


def test_auto_swipes_are_never_training_labels(conn):
    p = db.add_profile(conn, "tinder", ["a.jpg"])
    db.add_rating(conn, p, True, origin="auto")
    assert db.latest_ratings(conn) == {}                   # not a label
    assert db.auto_swiped_ids(conn) == {p: True}
    assert db.unrated_profile_ids(conn) == []              # but it is already handled in the app
    db.add_rating(conn, p, False, origin="app")            # a later human decision becomes the label
    assert db.latest_ratings(conn) == {p: False}
    assert db.auto_swiped_ids(conn) == {p: True}


def test_clear_and_undo_leave_auto_swipes_alone(conn):
    a = db.add_profile(conn, "tinder", ["a.jpg"])
    b = db.add_profile(conn, "tinder", ["b.jpg"])
    db.add_rating(conn, a, True, origin="manual")
    db.add_rating(conn, b, True, origin="auto")
    assert db.undo_last_rating(conn) == a                  # skips the newer auto rating
    assert db.undo_last_rating(conn) is None
    db.add_rating(conn, a, False, origin="app")
    assert db.clear_ratings(conn) == 1
    assert db.auto_swiped_ids(conn) == {b: True}


def test_auto_swipes_today_counts_only_auto_and_only_today(conn):
    a = db.add_profile(conn, "tinder", ["a.jpg"])
    db.add_rating(conn, a, True, origin="auto")
    db.add_rating(conn, a, True, origin="app")
    old = db.add_profile(conn, "tinder", ["b.jpg"])
    db.add_rating(conn, old, False, origin="auto")
    conn.execute("UPDATE ratings SET created_at = '2020-01-01 12:00:00' WHERE profile_id = ?", (old,))
    assert db.auto_swipes_today(conn) == 1
