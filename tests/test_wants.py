from datebot import db, wants


def conn_with(tmp_path, profiles):
    conn = db.connect(str(tmp_path / "t.db"))
    ids = [db.add_profile(conn, "tinder", f"x{i}", name=f"P{i}", bio=bio, extra=extra) for i, (bio, extra) in enumerate(profiles)]
    return conn, ids


def test_only_seeking_sentences_count_as_wants(tmp_path):
    conn, _ = conn_with(tmp_path, [
        ("Looking for a man who can make me laugh. I love dogs.", {}),
        ("I'm funny and I like hiking.", {}),  # about herself: not a want
        ("Söker en man med humor och värme", {}),
        ("Honestly, I like pizza", {}),  # 'honestly' is not 'honest'
    ])
    r = wants.build(conn)
    rows = {x["label"]: x for x in r["qualities"]}
    assert rows["humour / makes me laugh"]["n"] == 2
    assert "honest / open" not in rows
    assert rows["humour / makes me laugh"]["examples"][0]["text"].startswith("Looking for")


def test_avoid_needs_a_negation_and_prompts_are_tallied(tmp_path):
    prompts = {"prompts": ["What’s your biggest dealbreaker?: Never being on time", "What makes you laugh?: Memes"]}
    conn, ids = conn_with(tmp_path, [("No games please", {}), ("I love games", {}), *[("", prompts)] * 5])
    db.add_rating(conn, ids[0], liked=True)
    r = wants.build(conn)
    assert {x["label"]: x["n"] for x in r["avoid"]} == {"games / drama": 1}
    q = {c["question"]: c for c in r["prompts"]}
    assert q["What’s your biggest dealbreaker"]["n"] == 5
    assert q["What makes you laugh"]["answers"][0]["label"] == "Memes"
    assert r["avoid"][0]["rate"] == 1.0 and r["summary"]["rated"] == 1


def test_source_filter(tmp_path):
    conn, _ = conn_with(tmp_path, [("Looking for humor", {})])
    assert wants.build(conn, "badoo")["summary"]["profiles"] == 0
    assert wants.build(conn, "tinder")["summary"]["profiles"] == 1
