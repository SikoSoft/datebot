import importlib
import json
from types import SimpleNamespace

from datebot import db
from datebot.capture import hinge
from datebot.capture.recorder import Recorder
from datebot.interactions import InteractionTracker
from test_hinge import contents, users
from test_recorder import png


def flow(path, response, request="{}", t_req=0.0, t_end=0.0):
    return SimpleNamespace(
        request=SimpleNamespace(pretty_host=hinge.HOST, path=path, timestamp_start=t_req,
                                get_text=lambda strict=False: request),
        response=SimpleNamespace(status_code=200, timestamp_end=t_end, get_text=lambda strict=False: response))


def test_swipe_dwell_comes_from_network_timestamps(tmp_path, monkeypatch):
    monkeypatch.setenv("DATEBOT_HOME", str(tmp_path))
    addon = importlib.import_module("datebot.capture.hinge_addon")
    path = str(tmp_path / "t.db")
    rec = Recorder(fetch=lambda u: png() + u.encode(), source="hinge", db_path=path,
                   tracker=InteractionTracker("hinge", "iphone-proxy"))
    cap = addon.HingeCapture(rec)
    cap.response(flow(hinge.PATH_USER, json.dumps(users("u1", "u2")), t_end=100.0))
    cap.response(flow(hinge.PATH_CONTENT, json.dumps(contents("u1", "u2")), t_end=100.5))
    cap.response(flow(hinge.PATH_RATE, "{}", json.dumps({"subjectId": "u1", "rating": "skip"}), t_req=108.5))
    cap.response(flow(hinge.PATH_RATE, "{}", json.dumps({"subjectId": "u2", "rating": "note"}), t_req=121.0))
    cap.done()
    rows = db.interactions(db.connect(path))
    dwell = [json.loads(r["features"])["dwell_ms"] for r in rows]
    assert dwell == [8000, 12500]  # from arrival (100.5), then from the previous swipe (108.5)
    assert [r["liked"] for r in rows] == [0, 1] and all(r["profile_id"] for r in rows)
