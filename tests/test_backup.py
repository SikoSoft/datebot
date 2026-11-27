from PIL import Image
from typer.testing import CliRunner

from datebot import backup, db
from datebot.cli import app

runner = CliRunner()


def _seed(tmp_path, monkeypatch):
    monkeypatch.setenv("DATEBOT_HOME", str(tmp_path / "home"))
    src = tmp_path / "in"
    (src / "ann").mkdir(parents=True)
    Image.new("RGB", (8, 8), "red").save(src / "ann" / "1.png")
    runner.invoke(app, ["import", str(src)])
    with db.session() as conn:
        db.add_rating(conn, 1, True)


def test_backup_and_restore_roundtrip(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    out = runner.invoke(app, ["backup"]).output
    assert "1 profiles, 1 rating rows, 1 photo files" in out
    snap = backup.list_snapshots()[-1]

    with db.session() as conn:
        db.clear_ratings(conn)
    assert runner.invoke(app, ["restore", str(snap), "--yes"]).exit_code == 0
    with db.session() as conn:
        assert db.latest_ratings(conn) == {1: True}


def test_backup_refuses_nonempty_destination_and_no_photos(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    dest = backup.create(tmp_path / "snap", photos=False)
    assert not (dest / "photos").exists()
    try:
        backup.create(dest)
    except FileExistsError:
        pass
    else:
        raise AssertionError("should refuse to overwrite")


def test_reset_ratings_saves_a_backup_first(tmp_path, monkeypatch):
    _seed(tmp_path, monkeypatch)
    runner.invoke(app, ["reset-ratings", "--yes"])
    snap = backup.list_snapshots()[-1]
    assert snap.name.endswith("before-reset-ratings") and backup.verify(snap)["ratings"] == 1
