from datetime import date

from datebot.capture import tinder

# Shaped after community documentation of Tinder's web API; field names are ASSUMED, not verified
# against a live response. If real traffic differs, fix the parser and update this fixture.
RECS = {
    "meta": {"status": 200},
    "data": {"results": [
        {
            "type": "user",
            "distance_mi": 12,
            "user": {
                "_id": "abc123",
                "name": "Ann",
                "birth_date": "1996-06-15T00:00:00.000Z",
                "bio": "  Hiking and coffee  ",
                "city": {"name": "Portland"},
                "photos": [
                    {"url": "https://images.gotinder.com/abc/orig.jpg",
                     "processedFiles": [{"url": "https://x/172.jpg", "width": 172},
                                        {"url": "https://x/640.jpg", "width": 640}]},
                    {"url": "https://images.gotinder.com/abc/2.jpg"},
                ],
                "jobs": [{"title": {"name": "Nurse"}, "company": {"name": "City Hospital"}}],
                "schools": [{"name": "State U"}],
                "user_interests": {"selected_interests": []},
                "selected_descriptors": [{"name": "Smoking", "choice_selections": [{"name": "Non-smoker"}]}],
            },
        },
        {"type": "ad", "ad": {"id": "x"}},
        {"type": "user", "user": {"name": "no id"}},
    ]},
}


def test_parse_recs_extracts_profiles_and_skips_non_users():
    got = tinder.parse_recs(RECS)
    assert [p.external_id for p in got] == ["abc123"]
    p = got[0]
    assert (p.name, p.bio) == ("Ann", "Hiking and coffee")
    assert p.photo_urls == ["https://x/640.jpg", "https://images.gotinder.com/abc/2.jpg"]
    assert p.extra["distance_mi"] == 12 and p.extra["city"] == "Portland"
    assert p.extra["jobs"] == ["Nurse", "City Hospital"] and p.extra["schools"] == ["State U"]
    assert p.extra["descriptors"] == ["Smoking: Non-smoker"]
    assert "interests" not in p.extra


def test_parse_recs_tolerates_garbage():
    for bad in (None, {}, {"data": None}, {"data": {"results": "x"}}, {"data": {"results": [1, None]}}, []):
        assert tinder.parse_recs(bad) == []


def test_age_from_birth_date():
    today = date(2026, 6, 14)
    assert tinder.age_from_birth_date("1996-06-15T00:00:00.000Z", today) == 29  # birthday tomorrow
    assert tinder.age_from_birth_date("1996-06-14", today) == 30
    assert tinder.age_from_birth_date("garbage!!!", today) is None
    assert tinder.age_from_birth_date(None, today) is None


def test_parse_swipe_urls():
    like = tinder.parse_swipe("https://api.gotinder.com/like/abc123?locale=en&s_number=1")
    assert (like.user_id, like.liked) == ("abc123", True)
    assert tinder.parse_swipe("https://api.gotinder.com/like/abc123/super").liked is True
    assert tinder.parse_swipe("https://api.gotinder.com/pass/abc123").liked is False
    assert tinder.parse_swipe("https://api.gotinder.com/v2/pass/abc123").user_id == "abc123"
    assert tinder.parse_swipe("https://api.gotinder.com/v2/recs/core") is None
    assert tinder.parse_swipe("https://api.gotinder.com/v2/matches/like/abc") is None


def test_url_matchers():
    assert tinder.is_tinder_url("https://api.gotinder.com/x") and tinder.is_tinder_url("https://tinder.com/")
    assert not tinder.is_tinder_url("https://eviltinder.com/x")
    assert not tinder.is_tinder_url("https://example.com/?u=gotinder.com")
    assert tinder.is_recs_url("https://api.gotinder.com/v2/recs/core?locale=en")
    assert not tinder.is_recs_url("https://api.gotinder.com/v2/recs/other")
