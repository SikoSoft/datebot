import numpy as np
import pytest

from datebot import attributes, db, insights


def make_db(tmp_path):
    conn = db.connect(str(tmp_path / "t.db"))
    for i in range(40):
        pid = db.add_profile(conn, "tinder" if i % 2 else "badoo", f"x{i}", name=f"P{i}", age=25 + i % 10,
                             bio="I love my dog and travel" if i % 4 == 0 else None,
                             extra={"descriptors": [f"Workout: {'Often' if i < 20 else 'Never'}"] , "height_cm": 160 + i})
        db.add_rating(conn, pid, liked=i < 20)  # likes exactly the "Often" workouts
    db.add_rating(conn, 1, liked=False, origin="auto")  # the bot's swipes never count
    return conn


def test_wilson_bounds():
    lo, hi = insights.wilson(5, 10)
    assert 0 < lo < 0.5 < hi < 1
    assert insights.wilson(0, 0) == (0.0, 1.0)


def test_build_cards_and_highlights(tmp_path):
    conn = make_db(tmp_path)
    r = insights.build(conn, visual=None)
    assert r["summary"]["rated"] == 40 and r["summary"]["likes"] == 20 and r["summary"]["baseline"] == 0.5
    cards = {c["id"]: c for c in r["cards"]}
    rows = {x["label"]: x for x in cards["desc-workout"]["rows"]}
    assert rows["Often"]["rate"] == 1.0 and rows["Never"]["rate"] == 0.0 and rows["Often"]["n"] == 20
    assert {h["card"] for h in r["highlights"]} >= {"desc-workout"}
    assert "dogs" in {x["label"] for x in cards["themes"]["rows"]} or "themes" not in cards  # min_n=3 -> 10 profiles
    assert insights.build(conn, "tinder", visual=None)["summary"]["rated"] == 20


def test_ordered_buckets(tmp_path):
    r = insights.build(make_db(tmp_path), visual=None)
    age = next(c for c in r["cards"] if c["id"] == "age")
    assert [x["label"] for x in age["rows"]] == ["25–29", "30–34"] and age["ordered"]


def test_visual_cards_and_missing_embeddings(tmp_path):
    conn = make_db(tmp_path)
    labels = {pid: {"glasses": "yes" if pid <= 20 else "no", "hair_colour": "blonde"} for pid in range(1, 38)}
    r = insights.build(conn, visual=lambda c: labels)
    cards = {c["id"]: c for c in r["cards"]}
    assert cards["vis-glasses"]["visual"] and cards["vis-glasses"]["covered"] == 37
    assert {x["label"] for x in cards["vis-glasses"]["rows"]} == {"shows it", "doesn't stand out"}
    assert "3 rated profiles have no embedded photos" in r["summary"]["visual_note"]
    assert all(h["label"] != "doesn't stand out" for h in r["highlights"])


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def test_classify_categories_and_flags():
    rng = np.random.default_rng(0)
    d = 8
    prompts = attributes.all_prompts()
    pe = {p: unit(rng.normal(size=d)) for p in prompts}
    blonde = pe["a photo of a woman with blonde hair"]
    black = pe["a photo of a woman with black hair"]
    glasses = pe["a photo of a woman wearing prescription glasses"]
    person = unit(np.mean([pe[p] for p in attributes.GATES["person"][0]], axis=0))
    emb = [unit(person + 2 * blonde), unit(person + 2 * black), unit(person + 4 * glasses)] + [unit(person + rng.normal(size=d) * .2) for _ in range(40)]
    pids = list(range(len(emb)))
    groups = [attributes.GROUP_BY_KEY["hair_colour"], attributes.GROUP_BY_KEY["glasses"]]
    out = attributes.classify(pids, np.array(emb), pe, groups)
    assert out[0]["hair_colour"] == "blonde" and out[1]["hair_colour"] == "black"
    assert out[2]["glasses"] == "yes" and out[0]["glasses"] == "no"
