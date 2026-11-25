from PIL import Image
from typer.testing import CliRunner

from datebot.cli import app

runner = CliRunner()


def test_import_stats_and_reset_flow(tmp_path, monkeypatch):
    monkeypatch.setenv("DATEBOT_HOME", str(tmp_path / "home"))
    src = tmp_path / "in"
    for name in ("ann", "bea"):
        (src / name).mkdir(parents=True)
        Image.new("RGB", (8, 8), "red").save(src / name / "1.png")

    assert "Imported 2" in runner.invoke(app, ["import", str(src)]).output
    assert "2 profiles, 0 rated by you" in runner.invoke(app, ["stats"]).output
    assert "No model" in runner.invoke(app, ["reset-model"]).output
    assert "Deleted 0" in runner.invoke(app, ["reset-ratings", "--yes"]).output


def test_filters_set_show_clear(tmp_path, monkeypatch):
    monkeypatch.setenv("DATEBOT_HOME", str(tmp_path))
    assert "No filters" in runner.invoke(app, ["filters", "show"]).output
    out = runner.invoke(app, ["filters", "set", "--min-age", "25", "--block", "smoker", "--block", "420"]).output
    assert "Saved" in out
    shown = runner.invoke(app, ["filters", "show"]).output
    assert "min_age: 25" in shown and "smoker" in shown
    assert runner.invoke(app, ["filters", "set", "--min-age", "50", "--max-age", "30"]).exit_code == 1
    runner.invoke(app, ["filters", "clear"])
    assert "No filters" in runner.invoke(app, ["filters", "show"]).output


def test_capture_tinder_is_registered():
    out = runner.invoke(app, ["capture", "tinder", "--help"]).output
    assert "--cdp" in out and "--dump" in out


def test_eval_reports_and_explains_when_there_is_nothing_to_evaluate(tmp_path, monkeypatch):
    import numpy as np

    from datebot import db
    from datebot.embed import EMBED_ID

    monkeypatch.setenv("DATEBOT_HOME", str(tmp_path))
    empty = runner.invoke(app, ["eval"])
    assert empty.exit_code == 1 and "No rated profiles" in empty.output

    rng = np.random.default_rng(0)
    with db.session() as conn:
        for i in range(40):
            liked = i % 2 == 0
            pid = db.add_profile(conn, "manual", [f"{i}.jpg"], name=f"P{i}")
            photo_id = conn.execute("SELECT id FROM photos WHERE profile_id=?", (pid,)).fetchone()[0]
            db.set_embedding(conn, photo_id, (1 if liked else -1) + rng.normal(0, 1.5, 8), EMBED_ID)
            db.add_rating(conn, pid, liked)
    out = runner.invoke(app, ["eval"])
    assert out.exit_code == 0
    assert "Ratings: 40 (20 likes, 20 passes" in out.output and "ROC AUC" in out.output
    assert "Biggest surprises" in out.output


def test_autoswipe_refuses_without_a_model(tmp_path, monkeypatch):
    monkeypatch.setenv("DATEBOT_HOME", str(tmp_path))
    r = runner.invoke(app, ["autoswipe", "tinder", "--live"])
    assert r.exit_code == 1 and "No trained model" in r.output


def test_autoswipe_badoo_exists_and_refuses_without_a_model(tmp_path, monkeypatch):
    monkeypatch.setenv("DATEBOT_HOME", str(tmp_path))
    r = runner.invoke(app, ["autoswipe", "badoo", "--live"])
    assert r.exit_code == 1 and "No trained model" in r.output
    assert "--like-key" in runner.invoke(app, ["autoswipe", "badoo", "--help"]).output


def test_explain_reports_attributes_from_stored_embeddings(tmp_path, monkeypatch):
    import numpy as np

    from datebot import db, embed
    from datebot.embed import EMBED_ID

    monkeypatch.setenv("DATEBOT_HOME", str(tmp_path))
    assert "No embedded photos" in runner.invoke(app, ["explain"]).output
    rng = np.random.default_rng(0)
    with db.session() as conn:
        for i in range(40):
            liked = i % 2 == 0
            pid = db.add_profile(conn, "manual", [f"{i}.jpg"], name=f"P{i}")
            photo_id = conn.execute("SELECT id FROM photos WHERE profile_id=?", (pid,)).fetchone()[0]
            v = rng.normal(0, 0.3, 8)
            v[0] += 2 * liked
            db.set_embedding(conn, photo_id, v / np.linalg.norm(v), EMBED_ID)
            db.add_rating(conn, pid, liked)
    monkeypatch.setattr(embed, "embed_texts", lambda texts: np.eye(8)[[0] * len(texts)])
    out = runner.invoke(app, ["explain", "--group", "styling", "--prompt", "a red dress"])
    assert out.exit_code == 0, out.output
    assert "glasses" in out.output and "a red dress" in out.output and "AUC" in out.output
    assert runner.invoke(app, ["explain", "--group", "nonsense"]).exit_code == 1


def _fake_clip(monkeypatch):
    """Stand-in for CLIP so the CLI tests need no model download."""
    import numpy as np

    from datebot import embed

    calls = []

    def fake_images(paths, batch_size=16):
        calls.append(len(paths))
        rng = np.random.default_rng(len(calls))
        v = rng.normal(0, 1, (len(paths), 8)).astype(np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    monkeypatch.setattr(embed, "embed_images", fake_images)
    monkeypatch.setattr(embed, "embed_texts", lambda texts: np.eye(8, dtype=np.float32)[[0] * len(texts)])
    return calls


def _import_rated(tmp_path, n=10):
    from datebot import db

    src = tmp_path / "in"
    for i in range(n):
        (src / f"p{i}").mkdir(parents=True)
        Image.new("RGB", (8, 8), "red").save(src / f"p{i}" / "1.png")
    runner.invoke(app, ["import", str(src)])
    with db.session() as conn:
        for i, pid in enumerate(db.unrated_profile_ids(conn)):
            db.add_rating(conn, pid, i % 2 == 0)


def test_stats_flags_unembedded_profiles_and_train_embeds_them_itself(tmp_path, monkeypatch):
    monkeypatch.setenv("DATEBOT_HOME", str(tmp_path / "home"))
    calls = _fake_clip(monkeypatch)
    _import_rated(tmp_path)
    assert "10 not embedded yet" in runner.invoke(app, ["stats"]).output

    out = runner.invoke(app, ["train"])
    assert out.exit_code == 0, out.output
    assert "Embedded 10 new photo(s) first." in out.output and "Trained 'default' on 10 ratings" in out.output
    assert "not embedded" not in runner.invoke(app, ["stats"]).output

    runner.invoke(app, ["train"])
    assert calls == [10]  # nothing new, so CLIP was not run again


def test_eval_explain_and_score_embed_new_profiles_too(tmp_path, monkeypatch):
    monkeypatch.setenv("DATEBOT_HOME", str(tmp_path / "home"))
    calls = _fake_clip(monkeypatch)
    _import_rated(tmp_path, n=12)
    assert "Embedded 12 new photo(s) first." in runner.invoke(app, ["eval"]).output
    from datebot import db

    with db.session() as conn:
        db.add_profile(conn, "manual", ["new.jpg"], name="New")
    assert "Embedded 1 new photo(s) first." in runner.invoke(app, ["explain", "--group", "styling"]).output
    runner.invoke(app, ["train"])
    with db.session() as conn:
        db.add_profile(conn, "manual", ["newer.jpg"], name="Newer")
    out = runner.invoke(app, ["score"])
    assert out.exit_code == 0 and "Embedded 1 new photo(s) first." in out.output
    assert calls == [12, 1, 1]


def test_behavior_commands(tmp_path, monkeypatch):
    monkeypatch.setenv("DATEBOT_HOME", str(tmp_path))
    assert "No behaviour model" in runner.invoke(app, ["behavior", "reset"]).output
    assert runner.invoke(app, ["behavior", "show"]).exit_code == 1
    assert runner.invoke(app, ["behavior", "train", "--edges", "50,10"]).exit_code != 0
    r = runner.invoke(app, ["behavior", "train"])
    assert r.exit_code == 1 and "at least" in r.output
    assert "Deleted 0" in runner.invoke(app, ["behavior", "clear-data", "--yes"]).output
