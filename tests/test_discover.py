import json

from datebot.capture import discover


def test_worth_saving_only_badoo_api_like_responses():
    h = discover.BADOO_HOSTS
    assert discover.worth_saving("https://badoo.com/webapi", "application/json; charset=utf-8", h)
    assert discover.worth_saving("https://eu1.badoo.com/x", "text/plain", h)
    assert not discover.worth_saving("https://badoo.com/", "text/html", h)
    assert not discover.worth_saving("https://badoo.com/a.js", "application/javascript", h)
    assert not discover.worth_saving("https://badoo.com/p.jpg", "image/jpeg", h)
    assert not discover.worth_saving("https://notbadoo.com/x", "application/json", h)
    assert not discover.worth_saving("https://example.com/x", "application/json", h)


def test_save_and_summarize(tmp_path):
    out = tmp_path / "d"
    for i in range(3):
        discover.save_exchange(out, i + 1, "POST", "https://badoo.com/webapi?SERVER_X", 200, '{"a":1}', '{"b":2}')
    discover.save_exchange(out, 4, "GET", "https://badoo.com/other", 200, None, "x" * 50)
    d = json.loads((out / "00001-POST.json").read_text())
    assert d["request_body"] == '{"a":1}' and "headers" not in d and "cookies" not in d
    text = discover.summarize(out)
    assert "4 exchange(s)" in text and "POST /webapi?SERVER_X" in text and "GET /other" in text
    assert discover.summarize(tmp_path / "none") == "Nothing saved."


def test_oversized_response_is_truncated(tmp_path):
    p = discover.save_exchange(tmp_path, 1, "GET", "https://badoo.com/x", 200, None, "y" * (discover.MAX_BODY + 10))
    assert len(json.loads(p.read_text())["response_body"]) == discover.MAX_BODY
