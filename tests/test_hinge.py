"""Synthetic fixtures in the wire shapes verified from real Hinge traffic (no real profile data)."""
import json

from datebot.capture import hinge

PROMPTS = {"prompts": [{"id": "q1", "prompt": "A shower thought I recently had"},
                       {"id": "q2", "prompt": "Typical Sunday"}]}


def users(*ids):
    return [{"userId": i, "profile": {
        "age": 31, "firstName": "Sam", "lastName": "", "height": 170, "location": {"name": "Malmö"},
        "works": "Acme", "jobTitle": "Nurse", "educations": ["", "Lund"], "hometown": "Ystad", "drinking": 2}}
        for i in ids]


def contents(*ids):
    return [{"userId": i, "content": {
        "photos": [{"url": f"https://media.example/{i}/{n}.jpg", "caption": ""} for n in range(3)],
        "answers": [
            {"position": 1, "questionId": "q2", "type": "text", "response": "Brunch"},
            {"position": 0, "questionId": "q1", "type": "text", "response": "Socks are shoes"},
            {"position": 2, "questionId": "q1", "type": "voice", "response": ""}]}} for i in ids]


def session_with_profile(uid="u1"):
    s = hinge.HingeSession()
    s.handle(hinge.PATH_PROMPTS, 200, None, json.dumps(PROMPTS))
    assert s.handle(hinge.PATH_USER, 200, "{}", json.dumps(users(uid))) == []  # waits for the other half
    return s, s.handle(hinge.PATH_CONTENT, 200, "{}", json.dumps(contents(uid)))


def test_profile_assembled_from_both_responses():
    _, events = session_with_profile()
    p = events[0].profile
    assert (p.external_id, p.name, p.age, p.bio) == ("u1", "Sam", 31, None)
    assert p.photo_urls == [f"https://media.example/u1/{n}.jpg" for n in range(3)]
    assert p.extra["prompts"] == ["A shower thought I recently had: Socks are shoes", "Typical Sunday: Brunch"]
    assert p.extra["height_cm"] == 170 and p.extra["city"] == "Malmö"
    assert p.extra["descriptors"] == ["Job title: Nurse", "Work: Acme", "Hometown: Ystad", "Education: Lund"]


def test_content_may_arrive_before_user():
    s = hinge.HingeSession()
    assert s.handle(hinge.PATH_CONTENT, 200, "{}", json.dumps(contents("u2"))) == []
    assert s.handle(hinge.PATH_USER, 200, "{}", json.dumps(users("u2")))[0].profile.external_id == "u2"


def test_unknown_question_falls_back_to_answer_only():
    s = hinge.HingeSession()
    s.handle(hinge.PATH_USER, 200, "{}", json.dumps(users("u3")))
    p = s.handle(hinge.PATH_CONTENT, 200, "{}", json.dumps(contents("u3")))[0].profile
    assert p.extra["prompts"] == ["Socks are shoes", "Brunch"]


def test_swipes_map_note_to_like_and_skip_to_pass():
    s = hinge.HingeSession()
    like = s.handle(hinge.PATH_RATE, 200, json.dumps({"subjectId": "u1", "rating": "note"}), "{}")[0].swipe
    skip = s.handle(hinge.PATH_RATE, 200, json.dumps({"subjectId": "u2", "rating": "skip"}), "{}")[0].swipe
    assert (like.user_id, like.liked, skip.user_id, skip.liked) == ("u1", True, "u2", False)


def test_unverified_ratings_and_failed_calls_are_not_guessed():
    s = hinge.HingeSession()
    assert s.handle(hinge.PATH_RATE, 200, json.dumps({"subjectId": "u1", "rating": "rose"}), "{}") == []
    assert s.handle(hinge.PATH_RATE, 429, json.dumps({"subjectId": "u1", "rating": "note"}), "{}") == []
    assert dict(s.unknown_ratings) == {"rose": 1}


def test_garbage_is_ignored():
    s = hinge.HingeSession()
    assert s.handle(hinge.PATH_USER, 200, None, "not json") == []
    assert s.handle("/other", 200, None, "{}") == []


def test_replay_reads_a_probe_dump_in_order(tmp_path):
    def dump(n, path, req, res):
        (tmp_path / f"{n:05d}-POST.json").write_text(json.dumps(
            {"method": "POST", "url": f"https://{hinge.HOST}{path}", "status": 200,
             "request_body": req, "response_body": json.dumps(res)}))
    dump(1, hinge.PATH_PROMPTS, None, PROMPTS)
    dump(2, hinge.PATH_USER, "{}", users("u1"))
    dump(3, hinge.PATH_CONTENT, "{}", contents("u1"))
    dump(4, hinge.PATH_RATE, json.dumps({"subjectId": "u1", "rating": "skip"}), {"limit": None})
    kinds = [e.kind for e in hinge.replay(tmp_path)]
    assert kinds == ["profile", "swipe"]
