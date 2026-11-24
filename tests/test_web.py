import io

import pytest
from PIL import Image

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from datebot.web.app import create_app  # noqa: E402


def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "blue").save(buf, "PNG")
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
    for age in (19, 30):
        client.post("/api/profiles", files=[("photos", ("a.png", png(), "image/png"))], data={"age": age})
    assert client.get("/api/filters").json()["min_age"] is None

    r = client.put("/api/filters", json={"min_age": 21}).json()
    assert r["stats"]["hidden"] == 1
    assert client.get("/api/next").json()["profile"]["age"] == 30

    assert client.put("/api/filters", json={"min_age": 40, "max_age": 30}).status_code == 400
    assert client.put("/api/filters", json={}).json()["stats"]["hidden"] == 0
