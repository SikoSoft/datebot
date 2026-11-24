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
    assert "2 profiles, 0 rated" in runner.invoke(app, ["stats"]).output
    assert "No model" in runner.invoke(app, ["reset-model"]).output
    assert "Deleted 0" in runner.invoke(app, ["reset-ratings", "--yes"]).output
