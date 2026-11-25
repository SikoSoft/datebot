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


def test_height_rejects_at_or_above_threshold():
    f = Filters(reject_height_from_cm=178)
    assert check(prof(extra={"descriptors": ["Height: 178 cm"]}), f)
    assert check(prof(extra={"descriptors": ["Height: 185 cm"]}), f)
    assert check(prof(extra={"descriptors": ["Height: 177 cm"]}), f) is None
    assert check(prof(extra={"height_cm": 190}), f)
    assert check(prof(), f) is None  # unknown height never rejects


def test_field_keyword_only_matches_inside_its_field():
    f = Filters(field_keywords={"politics": ["left", "vänster"]})
    assert check(prof(extra={"descriptors": ["Politics: Left"]}), f)
    assert check(prof(extra={"descriptors": ["Politics: Vänster"]}), f)
    assert check(prof(extra={"descriptors": ["Politics: Liberal"]}), f) is None
    assert check(prof(bio="I left my heart in Rome"), f) is None


def test_phrase_keyword_for_hjartat_till_vanster():
    f = Filters(blocked_keywords=["hjärtat till vänster"])
    assert check(prof(bio="Med hjärtat till vänster"), f)


def test_rule_failing_rated_profiles_are_excluded_from_training(tmp_path, monkeypatch):
    from datebot import model
    monkeypatch.setenv("DATEBOT_HOME", str(tmp_path))
    filters.save(Filters(reject_height_from_cm=178))
    import numpy as np
    with db.session() as conn:
        pids = []
        for h in (190, 170):
            pid = db.add_profile(conn, "manual", ["x.jpg"], extra={"descriptors": [f"Height: {h} cm"]})
            (photo_id,) = [r["id"] for r in conn.execute("SELECT id FROM photos WHERE profile_id = ?", (pid,))]
            db.set_embedding(conn, photo_id, np.ones(4, dtype=np.float32), "m")
            db.add_rating(conn, pid, False)
            pids.append(pid)
        ids, _, _ = model.training_data(conn, "m")
    ok = pids[1]
    assert ids == [ok]
