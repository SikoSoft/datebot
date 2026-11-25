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


def test_merge_variants_keeps_the_more_common_spelling():
    merged = insights.merge_variants({"Malmö": [True] * 5, "Malmo": [False] * 2, " malmo": [True], "Lund": [True]})
    assert set(merged) == {"Malmö", "Lund"} and len(merged["Malmö"]) == 8


def test_city_card_merges_spellings(tmp_path):
    conn = db.connect(str(tmp_path / "c.db"))
    for i, city in enumerate(["Malmö"] * 5 + ["Malmo"] * 3 + ["Lund"] * 4):
        pid = db.add_profile(conn, "tinder", f"c{i}", extra={"city": city})
        db.add_rating(conn, pid, liked=i % 2 == 0)
    city = next(c for c in insights.build(conn, visual=None)["cards"] if c["id"] == "city")
    assert {r["label"]: r["n"] for r in city["rows"]} == {"Malmö": 8, "Lund": 4}


def test_fake_score_signals():
    from datebot import fake
    scam = db.Profile(id=1, source="tinder", external_id="a", name="A", age=27, bio="Crypto trader, text me on telegram",
                      extra={})
    real = db.Profile(id=2, source="tinder", external_id="b", name="B", age=27,
                      bio="I love hiking with my dog and baking sourdough on rainy Sundays", extra={})
    s = fake.score(scam, photo_count=1)
    assert s.score >= 55 and any("crypto" in r or "telegram" in r for r in s.reasons)
    assert fake.score(real, photo_count=5).score == 0
    assert fake.score(real, 5, {"professional": "yes", "filtered": "yes"}).score == 25


def test_fake_threshold_excludes_from_stats(tmp_path):
    conn = make_db(tmp_path)
    scam = db.add_profile(conn, "tinder", "scam", name="S", age=30, bio=None, extra={})
    db.add_rating(conn, scam, liked=True)
    base = insights.build(conn, visual=None)
    assert base["summary"]["rated"] == 41 and base["summary"]["fake"]["threshold"] is None
    r = insights.build(conn, visual=None, fake_threshold=25)
    f = r["summary"]["fake"]
    assert r["summary"]["rated"] == 40 - (f["excluded"] - 1)
    assert any(e["id"] == scam and e["liked"] for e in f["examples"])
    assert insights.build(conn, visual=None, fake_threshold=100)["summary"]["rated"] == 41


def test_manual_fake_marks_override_heuristic(tmp_path):
    conn = make_db(tmp_path)
    scam = db.add_profile(conn, "tinder", "scam", name="S", age=30, bio=None, extra={})
    db.add_rating(conn, scam, liked=True)
    real = db.add_profile(conn, "tinder", "real", name="R", age=30, bio="Hiking, dogs and sourdough on weekends", extra={})
    db.add_rating(conn, real, liked=True)
    db.set_fake_mark(conn, real, True)   # you say fake: excluded although the heuristic scores 0
    db.set_fake_mark(conn, scam, False)  # you say real: kept although the heuristic flags it
    r = insights.build(conn, visual=None, fake_threshold=25)
    ids = {e["id"] for e in r["summary"]["fake"]["examples"]}
    assert real in ids and scam not in ids
    assert next(e for e in r["summary"]["fake"]["examples"] if e["id"] == real)["manual"] is True
    assert insights.build(conn, visual=None)["summary"]["rated"] == 42  # slider off keeps everyone
    db.set_fake_mark(conn, real, None)
    assert db.fake_marks(conn) == {scam: False}
