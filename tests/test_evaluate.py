import numpy as np
import pytest

from datebot import evaluate, model


def make_data(n=60, signal=1.0, d=8, seed=0, text_signal=0.0, text_frac=1.0):
    """Features in the real layout [mean(d), max(d), text(d), has_text(1)] with a controllable signal."""
    rng = np.random.default_rng(seed)
    y = np.array([0, 1] * (n // 2))
    rng.shuffle(y)
    sign = (2 * y - 1)[:, None]
    photo = rng.normal(0, 1, (n, d)) + signal * sign
    X = []
    for i in range(n):
        has = rng.random() < text_frac
        text = rng.normal(0, 1, d) + text_signal * sign[i] if has else None
        X.append(model.featurize(np.stack([photo[i], photo[i]]), text))
    return np.stack(X), y


def test_oof_scores_cover_every_profile_and_beat_chance_with_signal():
    X, y = make_data(signal=1.5)
    oof = evaluate.oof_scores(X, y)
    assert oof.shape == y.shape and ((0 <= oof) & (oof <= 1)).all()
    h = evaluate.headline(y, oof)
    assert h.auc > 0.9 and h.accuracy > h.majority_baseline


def test_pure_noise_is_near_chance():
    X, y = make_data(signal=0.0, n=80)
    assert 0.3 < evaluate.headline(y, evaluate.oof_scores(X, y)).auc < 0.7


def test_refuses_too_few_labels():
    X, y = make_data(n=10)
    y[:] = 0
    y[0] = 1
    with pytest.raises(ValueError, match="at least"):
        evaluate.oof_scores(X, y)


def test_safe_auc_single_class_is_none():
    assert evaluate.safe_auc(np.zeros(5, dtype=int), np.random.rand(5)) is None


def test_ablation_attributes_signal_to_the_right_block():
    X, y = make_data(signal=0.0, text_signal=1.5, n=80)  # only the bio is informative
    a = evaluate.ablation(X, y)
    assert a["text only"] > 0.85 and a["photos + text"] > 0.85 and a["photos only"] < 0.7

    X, y = make_data(signal=1.5, text_signal=0.0, n=80)  # only photos are informative
    a = evaluate.ablation(X, y)
    assert a["photos only"] > 0.85 and a["text only"] < 0.7


def test_ablation_skips_text_when_nobody_has_a_bio():
    X, y = make_data(text_frac=0.0)
    assert evaluate.ablation(X, y)["text only"] is None
