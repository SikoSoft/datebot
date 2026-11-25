import numpy as np
import pytest

from datebot import explain
from datebot.explain import Attribute


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def make(n=90, d=16, seed=0):
    """Profiles with 2 photos each. Axis 0 is 'glasses' (drives likes), axis 1 is 'gym' (unrelated to likes)."""
    rng = np.random.default_rng(seed)
    liked = np.array([1, 0] * (n // 2))
    emb, pids, paths = [], [], []
    for pid in range(n):
        for k in range(2):
            v = rng.normal(0, 0.3, d)
            v[0] += 2.0 * liked[pid] * (k == 0)
            v[1] += 2.0 * rng.integers(0, 2)
            emb.append(v)
            pids.append(pid)
            paths.append(f"{pid}-{k}.jpg")
    text = np.eye(d)[:2]  # prompt 0 = glasses axis, prompt 1 = gym axis
    attrs = [Attribute("glasses", "g", "t"), Attribute("gym", "y", "t")]
    return pids, paths, unit(np.array(emb)), unit(text), attrs, {pid: bool(liked[pid]) for pid in range(n)}


def test_finds_the_attribute_that_drives_likes_and_not_the_unrelated_one():
    pids, paths, emb, text, attrs, ratings = make()
    res = {r.attribute.name: r for r in explain.analyse(pids, paths, emb, text, attrs, ratings)}
    assert res["glasses"].auc > 0.85 and res["glasses"].significant and res["glasses"].verdict == "you like this"
    assert res["glasses"].like_rate_high > res["glasses"].like_rate_low
    assert 0.35 < res["gym"].auc < 0.65 and not res["gym"].significant
    assert explain.analyse(pids, paths, emb, text, attrs, ratings)[0].attribute.name == "glasses"  # strongest first


def test_detects_things_you_pass_on():
    pids, paths, emb, text, attrs, ratings = make()
    flipped = {pid: not liked for pid, liked in ratings.items()}
    r = explain.analyse(pids, paths, emb, text, attrs, flipped)[0]
    assert r.auc < 0.15 and r.verdict == "you tend to pass on this"


def test_examples_are_the_highest_scoring_photos():
    pids, paths, emb, text, attrs, ratings = make()
    r = next(r for r in explain.analyse(pids, paths, emb, text, attrs, ratings, examples=3) if r.attribute.name == "glasses")
    assert len(r.examples) == 3 and all(p.endswith("-0.jpg") for p in r.examples)


def test_unrated_profiles_are_ignored_and_too_little_data_is_refused():
    pids, paths, emb, text, attrs, ratings = make()
    few = {pid: v for pid, v in list(ratings.items())[:2]}
    with pytest.raises(ValueError, match="at least"):
        explain.analyse(pids, paths, emb, text, attrs, few)


def test_benjamini_hochberg_is_monotone_and_bounded():
    p = np.array([0.001, 0.04, 0.5, 0.03])
    q = explain.benjamini_hochberg(p)
    assert (q >= p).all() and (q <= 1).all() and q[0] == pytest.approx(0.004)


def test_report_warns_with_few_ratings_and_lists_attributes():
    pids, paths, emb, text, attrs, ratings = make()
    out = explain.format_report(explain.analyse(pids, paths, emb, text, attrs, ratings), 10, 20)
    assert "WARNING" in out and "glasses" in out and "false-discovery" in out
