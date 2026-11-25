import json

from datebot.capture import hinge_probe


def test_worth_saving_skips_media():
    assert hinge_probe.worth_saving("application/json")
    assert not hinge_probe.worth_saving("image/jpeg")


def test_summarize_reports_hosts_and_pinned(tmp_path):
    (tmp_path / "00001-GET.json").write_text(json.dumps(
        {"method": "GET", "url": "https://api.example.com/x/y", "response_body": "{}" * 10}))
    (tmp_path / "tls-failures.txt").write_text("10:00:00 pinned.example.com\n")
    out = hinge_probe.summarize(tmp_path)
    assert "api.example.com" in out and "pinned.example.com" in out


def test_summarize_handles_host_with_only_empty_responses(tmp_path):
    (tmp_path / "00001-GET.json").write_text(json.dumps(
        {"method": "GET", "url": "https://empty.example.com/x", "response_body": ""}))
    assert "empty.example.com" in hinge_probe.summarize(tmp_path)
