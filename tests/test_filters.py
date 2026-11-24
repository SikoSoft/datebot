import pytest

from datebot import db, filters
from datebot.filters import Filters, check


def prof(**kw):
    base = dict(id=1, source="manual", external_id=None, name="A", age=None, bio=None, extra={})
    return db.Profile(**{**base, **kw})


def test_empty_filters_pass_everything():
    assert check(prof(age=99, bio="anything"), Filters()) is None


def test_age_range_and_missing_age_passes():
    f = Filters(min_age=25, max_age=35)
    assert check(prof(age=24), f) and check(prof(age=36), f)
    assert check(prof(age=25), f) is None and check(prof(age=35), f) is None
    assert check(prof(age=None), f) is None


def test_distance_km_and_miles():
    f = Filters(max_distance_km=50)
    assert check(prof(extra={"distance_km": 51}), f)
    assert check(prof(extra={"distance_km": 50}), f) is None
    assert check(prof(extra={"distance_mi": 40}), f)  # ~64 km
    assert check(prof(extra={"distance_mi": 10}), f) is None
    assert check(prof(), f) is None


def test_blocked_keywords_match_whole_words_incl_prompts():
    f = Filters(blocked_keywords=["smoker", "420"])
    assert check(prof(bio="Non-smoker here"), f) is None  # hyphenated words are not split
    assert check(prof(bio="heavy smoker"), f)
    assert check(prof(bio="4200 miles away"), f) is None
    assert check(prof(extra={"prompts": [{"q": "x", "a": "I love 420 friendly"}]}), f)


def test_required_keywords_need_any_match():
    f = Filters(required_keywords=["hiking", "climbing"])
    assert check(prof(bio="I like Hiking!"), f) is None
    assert check(prof(bio="I like tv"), f)
    assert check(prof(bio=None), f)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("DATEBOT_HOME", str(tmp_path))


def test_save_load_roundtrip(home):
    assert filters.load().is_empty()
    filters.save(Filters(min_age=21, blocked_keywords=["x"]))
    assert filters.load() == Filters(min_age=21, blocked_keywords=["x"])


def test_eligible_unrated_excludes_filtered_and_rated(home, tmp_path):
    conn = db.connect(str(tmp_path / "t.db"))
    young = db.add_profile(conn, "manual", ["a.jpg"], age=19)
    ok = db.add_profile(conn, "manual", ["b.jpg"], age=30)
    rated = db.add_profile(conn, "manual", ["c.jpg"], age=30)
    db.add_rating(conn, rated, True)
    assert filters.eligible_unrated(conn, Filters(min_age=21)) == [ok]
    assert filters.eligible_unrated(conn, Filters()) == [young, ok]
