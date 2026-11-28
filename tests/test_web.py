import io

import pytest
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from datebot.web.app import create_app  # noqa: E402


def png(color: str = "blue") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATEBOT_HOME", str(tmp_path))
    return TestClient(create_app())


def test_upload_rate_undo_reset(client):
    assert client.get("/api/next").json()["profile"] is None
    r = client.post("/api/profiles", files=[("photos", ("a.png", png(), "image/png"))], data={"name": "Ann"})
    pid = r.json()["id"]

    nxt = client.get("/api/next").json()
    assert nxt["profile"]["id"] == pid and nxt["profile"]["name"] == "Ann"
    assert client.get(nxt["profile"]["photos"][0]).status_code == 200

    assert client.post("/api/rate", json={"profile_id": pid, "liked": True}).json()["likes"] == 1
    assert client.get("/api/next").json()["profile"] is None
    assert client.post("/api/undo").json()["undone_profile_id"] == pid
    assert client.get("/api/next").json()["profile"]["id"] == pid

    client.post("/api/rate", json={"profile_id": pid, "liked": False})
    assert client.post("/api/reset-ratings").json()["deleted"] == 1


def test_rejects_non_images_and_unknown_profile(client):
    bad = client.post("/api/profiles", files=[("photos", ("a.txt", b"hi", "text/plain"))])
    assert bad.status_code == 400
    assert client.post("/api/rate", json={"profile_id": 99, "liked": True}).status_code == 404


def test_train_needs_enough_ratings(client):
    assert client.post("/api/train").status_code == 400


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Datebot" in r.text


def test_filters_hide_profiles_from_queue(client):
    for age, color in ((19, "red"), (30, "green")):
        client.post("/api/profiles", files=[("photos", ("a.png", png(color), "image/png"))], data={"age": age})
    assert client.get("/api/filters").json()["min_age"] is None

    r = client.put("/api/filters", json={"min_age": 21}).json()
    assert r["stats"]["hidden"] == 1
    assert client.get("/api/next").json()["profile"]["age"] == 30

    assert client.put("/api/filters", json={"min_age": 40, "max_age": 30}).status_code == 400
    assert client.put("/api/filters", json={"min_age": None, "max_age": None}).json()["stats"]["hidden"] == 0


def test_eval_endpoint_explains_when_not_ready(client):
    r = client.get("/api/eval")
    assert r.status_code == 400 and "No rated profiles" in r.json()["detail"]


def test_review_queue_audit_and_feedback(client):
    from datebot import db

    empty = client.get("/api/review").json()
    assert empty["auto"] == [] and empty["pending"] == [] and empty["summary"]["auto"] == 0

    ids = []
    for name, color in (("Ann", "red"), ("Bea", "green"), ("Cy", "yellow")):
        r = client.post("/api/profiles", files=[("photos", (f"{name}.png", png(color), "image/png"))], data={"name": name})
        ids.append(r.json()["id"])
    with db.session() as conn:
        db.add_rating(conn, ids[0], True, origin="auto")
        db.add_rating(conn, ids[1], False, origin="auto")

    r = client.get("/api/review").json()
    assert {a["name"]: a["bot_liked"] for a in r["auto"]} == {"Ann": True, "Bea": False}
    assert all(a["yours"] is None for a in r["auto"]) and r["summary"]["unreviewed"] == 2
    assert r["stats"]["rated"] == 0                       # the bot's swipes are not your ratings
    assert r["auto"][0]["name"] == "Bea"                  # newest first
    assert client.get(r["auto"][0]["photos"][0]).status_code == 200

    client.post("/api/rate", json={"profile_id": ids[0], "liked": False})   # you disagree with the like
    client.post("/api/rate", json={"profile_id": ids[1], "liked": False})   # you agree with the pass
    s = client.get("/api/review").json()["summary"]
    assert (s["unreviewed"], s["agreed"], s["disagreed"]) == (0, 1, 1)
    assert client.post("/api/undo").json()["undone_profile_id"] == ids[1]   # undo hits your rating, not the bot's
    assert client.get("/api/review").json()["summary"]["unreviewed"] == 1


def test_review_page_is_served(client):
    r = client.get("/review")
    assert r.status_code == 200 and "Review queue" in r.text
    assert "/static/nav.js" in client.get("/").text  # the nav (with the Review link) is shared, see nav.js


def test_saving_filters_from_the_panel_keeps_rules_it_cannot_edit(client):
    client.put("/api/filters", json={"reject_height_from_cm": 178, "field_keywords": {"politics": ["left"]}})
    r = client.put("/api/filters", json={"min_age": 21}).json()["filters"]
    assert r["min_age"] == 21 and r["reject_height_from_cm"] == 178 and r["field_keywords"] == {"politics": ["left"]}


def test_insights_endpoint_and_page(client):
    assert client.get("/insights").status_code == 200
    body = client.get("/api/insights").json()
    assert body["summary"]["rated"] == 0 and body["cards"] == []


def test_wants_endpoint_and_page(client):
    assert client.get("/wants").status_code == 200
    body = client.get("/api/wants").json()
    assert body["summary"]["profiles"] == 0 and body["qualities"] == [] and body["prompts"] == []


def test_fake_endpoint_marks_passes_and_undo_clears(client):
    pid = client.post("/api/profiles", files=[("photos", ("a.png", png(), "image/png"))]).json()["id"]
    assert client.post("/api/fake", json={"profile_id": pid, "is_fake": True, "and_pass": True}).json()["rated"] == 1
    assert client.get("/api/next").json()["profile"] is None
    assert client.post("/api/undo").status_code == 200
    assert client.get("/api/next").json()["profile"]["id"] == pid
    assert client.post("/api/fake", json={"profile_id": 999999, "is_fake": True}).status_code == 404


def test_browse_lists_profiles_without_rating_anything(client):
    assert client.get("/api/browse").json()["items"] == []
    r = client.post("/api/profiles", files=[("photos", ("a.png", png(), "image/png"))], data={"name": "Ann"})
    pid = r.json()["id"]

    b = client.get("/api/browse").json()
    assert [(i["name"], i["yours"], i["score"]) for i in b["items"]] == [("Ann", None, None)] and not b["model_active"]
    one = client.get(f"/api/browse/{pid}").json()
    assert one["profile"]["name"] == "Ann" and one["why"] is None
    assert client.get("/api/browse/999").status_code == 404
    assert client.get("/api/next").json()["stats"]["rated"] == 0   # browsing rated nothing
    assert client.get("/browse").status_code == 200


def test_reuploading_same_photos_returns_existing_profile(client):
    first = client.post("/api/profiles", files=[("photos", ("a.png", png(), "image/png"))], data={"name": "Ann"}).json()
    again = client.post("/api/profiles", files=[("photos", ("b.png", png(), "image/png"))]).json()
    other = client.post("/api/profiles", files=[("photos", ("c.png", png("red"), "image/png"))]).json()
    assert again["id"] == first["id"] and other["id"] != first["id"]


def test_ideal_endpoint_and_page(client):
    assert client.get("/ideal").status_code == 200
    body = client.get("/api/ideal").json()
    assert body["looks"] == [] and body["matches"] == [] and body["model_active"] is False


def test_stats_report_how_far_the_model_is_behind(client):
    assert client.get("/api/next").json()["stats"]["behind"] is None  # no model yet


def test_rating_triggers_background_retrain_only_when_a_model_is_far_enough_behind(client, monkeypatch):
    import threading

    from datebot import model
    from datebot.web import app as webapp

    ran = threading.Event()

    def fake_job(name):
        ran.set()
        webapp._retraining.release()

    monkeypatch.setattr(webapp, "_retrain_job", fake_job)
    monkeypatch.setattr(webapp, "AUTO_RETRAIN_EVERY", 2)
    monkeypatch.setattr(model, "load", lambda name="default": model.Trained(None, "m", 0, 0, None, "2000-01-01T00:00:00+00:00"))
    ids = [client.post("/api/profiles", files=[("photos", ("a.png", png(c), "image/png"))]).json()["id"]
           for c in ("red", "green", "blue")]
    client.post("/api/rate", json={"profile_id": ids[0], "liked": True})
    assert not ran.is_set()  # 1 behind, below the threshold
    s = client.post("/api/rate", json={"profile_id": ids[1], "liked": False}).json()
    assert ran.wait(2) and s["retrain_every"] == 2


def test_retrain_needs_a_model_and_runs_one_at_a_time(monkeypatch):
    from datebot.web import app as webapp

    started = []
    monkeypatch.setattr(webapp, "_retrain_job", lambda name: started.append(name))  # never releases the lock
    assert webapp.maybe_retrain(None) is False and webapp.maybe_retrain(webapp.AUTO_RETRAIN_EVERY - 1) is False
    assert webapp.maybe_retrain(webapp.AUTO_RETRAIN_EVERY) is True
    assert webapp.maybe_retrain(webapp.AUTO_RETRAIN_EVERY) is False  # one is already running
    webapp._retraining.release()


def test_stats_endpoint_reflects_ratings_made_elsewhere(client):
    from datebot import db

    pid = client.post("/api/profiles", files=[("photos", ("a.png", png(), "image/png"))], data={"name": "Ann"}).json()["id"]
    assert client.get("/api/stats").json()["rated"] == 0
    with db.session() as conn:  # what a capture session does: an 'app' rating written outside the UI
        db.add_rating(conn, pid, True, origin="app")
    s = client.get("/api/stats").json()
    assert (s["rated"], s["likes"]) == (1, 1)


def test_stats_says_when_the_current_profile_was_rated_elsewhere(client):
    from datebot import db

    pid = client.post("/api/profiles", files=[("photos", ("a.png", png(), "image/png"))], data={"name": "Ann"}).json()["id"]
    assert client.get(f"/api/stats?profile_id={pid}").json()["profile_rated"] is False
    with db.session() as conn:
        db.add_rating(conn, pid, False, origin="app")
    assert client.get(f"/api/stats?profile_id={pid}").json()["profile_rated"] is True
    assert "profile_rated" not in client.get("/api/stats").json()


def test_card_carries_saved_profile_details(client):
    from datebot import db
    with db.session() as conn:
        pid = db.add_profile(conn, "tinder", [], external_id="x", name="A", age=30, bio="hi",
                             extra={"city": "Malmö", "distance_mi": 5, "height_cm": 170, "jobs": ["Nurse"],
                                    "prompts": ["Q: A"], "interests": ["Yoga", "Hiking"]})
        card = __import__("datebot.web.app", fromlist=["_card"])._card(conn, pid, None)
    by = {d["label"]: d["lines"] for d in card["details"]}
    assert by["Location"] == ["Malmö · 8 km away"]
    assert by["Height"] == ["170 cm"] and by["Work"] == ["Nurse"] and by["Prompts"] == ["Q: A"]
    assert by["Interests"] == ["Yoga, Hiking"]


def test_behavior_page_and_untrained_endpoint(client):
    assert client.get("/behavior").status_code == 200
    body = client.get("/api/behavior").json()
    assert body["trained"] is False and body["interactions"] == 0
    assert client.post("/api/behavior/train").status_code == 400


def test_behavior_endpoint_lists_sources_and_accepts_a_source(client):
    assert client.get("/api/behavior", params={"name": "tinder"}).json()["sources"] == []
    assert client.post("/api/behavior/train", params={"name": "tinder", "source": "tinder"}).status_code == 400
