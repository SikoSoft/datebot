import io

import pytest
from PIL import Image

from datebot import db
from datebot.capture.recorder import Recorder
from datebot.capture.tinder import CapturedProfile, Swipe


def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "green").save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("DATEBOT_HOME", str(tmp_path))
    fetched = []

    def fetch(url):
        fetched.append(url)
        if "bad" in url:
            raise OSError("boom")
        return png() + url.encode()  # distinct bytes per url, so photos are not deduped together

    return Recorder(fetch=fetch, db_path=str(tmp_path / "t.db")), fetched, str(tmp_path / "t.db")


def prof(i="u1", urls=("https://x/a.jpg", "https://x/b.jpg"), **kw):
    return CapturedProfile(external_id=i, name="Ann", age=30, bio="hi", photo_urls=list(urls), **kw)


def test_stores_profile_with_ordered_photos_and_fields(env):
    rec, _, path = env
    assert rec.record_profiles([prof(extra={"city": "Portland"})]) == 1
    conn = db.connect(path)
    pid = db.find_profile_id(conn, "tinder", "u1")
    p = db.get_profile(conn, pid)
    assert (p.name, p.age, p.bio, p.extra) == ("Ann", 30, "hi", {"city": "Portland"})
    assert len(db.photo_paths(conn, pid)) == 2


def test_known_profiles_are_not_downloaded_again(env):
    rec, fetched, _ = env
    rec.record_profiles([prof()])
    n = len(fetched)
    assert rec.record_profiles([prof()]) == 0
    assert len(fetched) == n
    assert (rec.counts.seen, rec.counts.new) == (2, 1)


def test_failed_photo_is_skipped_but_profile_kept_if_any_succeed(env):
    rec, _, path = env
    rec.record_profiles([prof(urls=("https://x/bad.jpg", "https://x/ok.jpg"))])
    conn = db.connect(path)
    assert len(db.photo_paths(conn, db.find_profile_id(conn, "tinder", "u1"))) == 1


def test_profile_with_no_downloadable_photo_is_not_stored_and_can_retry(env):
    rec, _, path = env
    assert rec.record_profiles([prof(urls=("https://x/bad.jpg",))]) == 0
    assert rec.counts.failed == 1
    assert db.find_profile_id(db.connect(path), "tinder", "u1") is None
    assert rec.record_profiles([prof()]) == 1


def test_swipes_become_app_ratings_and_unknown_ones_are_counted(env):
    rec, _, path = env
    rec.record_profiles([prof("a"), prof("b", urls=("https://x/c.jpg",))])
    assert rec.record_swipe(Swipe("a", True)) and rec.record_swipe(Swipe("b", False))
    assert rec.record_swipe(Swipe("zzz", True)) is False
    conn = db.connect(path)
    ratings = db.latest_ratings(conn)
    assert ratings == {db.find_profile_id(conn, "tinder", "a"): True, db.find_profile_id(conn, "tinder", "b"): False}
    assert {r[0] for r in conn.execute("SELECT origin FROM ratings")} == {"app"}
    assert (rec.counts.swipes, rec.counts.unknown_swipes) == (2, 1)


def test_bot_swipes_are_recorded_with_auto_origin(env):
    rec, _, path = env
    rec.record_profiles([prof("a")])
    assert rec.record_swipe(Swipe("a", True), origin="auto")
    conn = db.connect(path)
    assert db.latest_ratings(conn) == {}
    assert db.auto_swiped_ids(conn) == {db.find_profile_id(conn, "tinder", "a"): True}


def test_first_app_swipe_can_be_skipped_once_and_bot_swipes_do_not_use_it_up(env):
    rec, _, _ = env
    rec._skip_pending = True
    assert rec.take_skip("auto") is False
    assert rec.take_skip("app") is True
    assert rec.take_skip("app") is False
    assert rec.counts.ignored == 1
