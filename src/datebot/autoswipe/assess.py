"""Glue between the database/model and the auto-swipe controller."""
from __future__ import annotations

from datebot import db, embed, filters, model
from datebot.autoswipe.logic import Bands


def bands_from_eval(conn, like_precision: float, pass_precision: float) -> Bands:
    """Cut-offs chosen from out-of-fold evaluation of your own ratings (see `datebot eval`)."""
    from datebot import evaluate

    t = evaluate.run(conn, embed.EMBED_ID, like_precision, pass_precision).thresholds
    return Bands(like_at=t.like_at, pass_at=t.pass_at)


def make_assess(trained, f: filters.Filters, source: str = "tinder"):
    """Return assess(external_id) -> (score, hard-filter reason), embedding the profile on first sight."""

    def assess(external_id: str):
        with db.session() as conn:
            pid = db.find_profile_id(conn, source, external_id)
            if pid is None:
                return None, None
            embed.embed_pending(conn)
            reason = filters.check(db.get_profile(conn, pid), f)
            return model.score_profile(conn, trained, pid), reason

    return assess
