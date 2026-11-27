from datebot.web.links import profile_url


def test_profile_url_per_source():
    assert profile_url("tinder", "abc") == "https://tinder.com/app/profile/abc"
    assert profile_url("badoo", "123") == "https://badoo.com/profile/123"


def test_no_link_without_template_or_id():
    assert profile_url("hinge", "abc") is None  # no web profile exists
    assert profile_url("manual", "abc") is None
    assert profile_url("tinder", None) is None
