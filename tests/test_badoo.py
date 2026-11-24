import json

import pytest

from datebot import db
from datebot.capture import badoo, badoo_browser
from datebot.capture.recorder import Recorder


def user(i, **kw):
    u = {"user_id": f"id{i}", "name": f"N{i}", "age": 30 + i, "city": {"id": 1, "name": "Malmo"},
         "albums": [{"photos": [{"id": "p1", "large_url": f"//cdn.example/p{i}a.jpg"},
                                {"id": "p2", "preview_url": f"//cdn.example/p{i}b.jpg"}]}],
         "interests": [{"name": "Hiking"}, {"name": "Shopping"}],
         "profile_fields": [
             {"id": "location", "type": 1, "name": "Current location", "display_value": "Malmo"},
             {"id": "aboutme_text", "type": 2, "name": "About me", "display_value": "Love dogs"},
             {"id": "smoking", "type": 8, "name": "Smoking", "display_value": "Non-smoker"},
             {"id": "229", "type": 26, "name": "Weekends?", "display_value": "Hiking"},
             {"id": "languages", "type": 11, "name": "You speak", "display_value": "Swedish"}],
         "tiw_idea": {"tiw_phrase": "Ready for a relationship"}}
    u.update(kw)
    return {"has_user_voted": False, "user": u}


def encounters(*items):
    return {"message_type": 6004, "body": [
        {"message_type": 486},
        {"message_type": 84, "client_encounters": {"results": list(items)}}]}


def vote(person, v):
    return json.dumps({"body": [{"message_type": 80, "server_encounters_vote": {"person_id": person, "vote": v}}]})


def test_api_name():
    assert badoo.api_name("https://euw4.badoo.com/mwebapi.phtml?SERVER_GET_ENCOUNTERS") == "SERVER_GET_ENCOUNTERS"
    assert badoo.api_name("https://badoo.com/other?SERVER_X") is None
    assert badoo.api_name("https://evil.com/mwebapi.phtml?SERVER_X") is None


def test_parse_encounters():
    [p] = badoo.parse_encounters(encounters(user(1)))
    assert (p.external_id, p.name, p.age, p.bio) == ("id1", "N1", 31, "Love dogs")
    assert p.photo_urls == ["https://cdn.example/p1a.jpg", "https://cdn.example/p1b.jpg"]
    assert p.extra["city"] == "Malmo" and p.extra["interests"] == ["Hiking", "Shopping"]
    assert p.extra["prompts"] == ["Weekends?: Hiking"]
    assert "Smoking: Non-smoker" in p.extra["descriptors"] and "Looking for: Ready for a relationship" in p.extra["descriptors"]
    assert not any("Swedish" in d for d in p.extra["descriptors"])


def test_parse_encounters_is_defensive():
    assert badoo.parse_encounters(None) == [] and badoo.parse_encounters("not json") == []
    assert badoo.parse_encounters({"body": [{"message_type": 84, "client_encounters": {}}]}) == []
    assert badoo.parse_encounters(encounters({"user": {"name": "no id"}}, "junk")) == []
    [p] = badoo.parse_encounters(json.dumps(encounters(user(2, age=0, albums=[], profile_fields=[]))))
    assert p.age is None and p.photo_urls == [] and p.bio is None


def test_bio_falls_back_to_displayed_about_me():
    [p] = badoo.parse_encounters(encounters(user(1, profile_fields=[], displayed_about_me=["", "Hello"])))
    assert p.bio == "Hello"


def test_parse_vote():
    assert badoo.parse_vote(vote("a", 2)).liked is True
    s = badoo.parse_vote(vote("a", 3))
    assert s.user_id == "a" and s.liked is False
    assert badoo.parse_vote(vote("a", 1)) is None and badoo.parse_vote(vote("a", 4)) is None
    assert badoo.parse_vote(None) is None and badoo.parse_vote("{") is None


class Req:
    method = "POST"

    def __init__(self, post): self.post_data = post


class Resp:
    ok = True

    def __init__(self, url, body, post=None):
        self.url, self._body, self.request = url, body, Req(post)

    def text(self): return self._body


class Ctx:
    def on(self, event, fn): self.fn = fn


def test_listener_records_profiles_and_swipes(tmp_path):
    rec = Recorder(fetch=lambda url: _png(), source="badoo", db_path=str(tmp_path / "t.db"))
    ctx, logs = Ctx(), []
    badoo_browser.attach(ctx, rec, log=logs.append)
    base = "https://euw4.badoo.com/mwebapi.phtml?"
    ctx.fn(Resp(base + "SERVER_GET_ENCOUNTERS", json.dumps(encounters(user(1), user(2)))))
    ctx.fn(Resp(base + "SERVER_ENCOUNTERS_VOTE", "{}", vote("id1", 2)))
    ctx.fn(Resp(base + "SERVER_ENCOUNTERS_VOTE", "{}", vote("id2", 3)))
    ctx.fn(Resp(base + "SERVER_ENCOUNTERS_VOTE", "{}", vote("nobody", 2)))
    ctx.fn(Resp(base + "SERVER_APP_STATS", "garbage"))
    conn = db.connect(str(tmp_path / "t.db"))
    a, b = db.find_profile_id(conn, "badoo", "id1"), db.find_profile_id(conn, "badoo", "id2")
    assert db.latest_ratings(conn) == {a: True, b: False}
    assert rec.counts.unknown_swipes == 1
    assert db.get_profile(conn, a).bio == "Love dogs"


def test_listener_survives_garbage(tmp_path):
    rec = Recorder(fetch=lambda url: b"", source="badoo", db_path=str(tmp_path / "t.db"))
    ctx, logs = Ctx(), []
    badoo_browser.attach(ctx, rec, log=logs.append)
    ctx.fn(Resp("https://badoo.com/mwebapi.phtml?SERVER_GET_ENCOUNTERS", "not json"))
    assert rec.counts.seen == 0


def _png():
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(buf, "PNG")
    return buf.getvalue()
