from __future__ import annotations

import tempfile
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from datebot import config, db, filters, ideal, ingest, insights, model

STATIC = Path(__file__).parent / "static"


class RateBody(BaseModel):
    profile_id: int
    liked: bool


class FakeBody(BaseModel):
    profile_id: int
    is_fake: bool | None  # null clears your mark
    and_pass: bool = False


class FiltersBody(BaseModel):
    min_age: int | None = None
    max_age: int | None = None
    max_distance_km: float | None = None
    blocked_keywords: list[str] = []
    required_keywords: list[str] = []
    reject_height_from_cm: int | None = None
    field_keywords: dict[str, list[str]] = {}


def _stats(conn) -> dict:
    ratings = db.latest_ratings(conn)
    total = conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
    likes = sum(ratings.values())
    unrated = len(db.unrated_profile_ids(conn))
    return {"total": total, "rated": len(ratings), "likes": likes, "passes": len(ratings) - likes,
            "auto_swiped": len(db.auto_swiped_ids(conn)), "unrated": unrated, "hidden": unrated - len(filters.eligible_unrated(conn)),
            "models": model.list_models(), "behind": model.ratings_behind(conn, model.load())}


def _card(conn, pid: int, score: float | None) -> dict:
    p = db.get_profile(conn, pid)
    return {"id": pid, "name": p.name, "age": p.age, "bio": p.bio, "source": p.source, "score": score,
            "photos": [f"/photos/{Path(x).name}" for x in db.photo_paths(conn, pid)]}


def create_app() -> FastAPI:
    app = FastAPI(title="datebot")
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")
    app.mount("/photos", StaticFiles(directory=str(config.photos_dir())), name="photos")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/next")
    def next_profile(model_name: str = "default"):
        """Next profile to rate: the model's most uncertain pick if a model exists, else oldest unrated."""
        with db.session() as conn:
            trained = model.load(model_name)
            eligible = filters.eligible_unrated(conn)  # hard filters: hidden profiles are never shown
            card = None
            if trained:
                picks = model.most_uncertain(conn, trained, 1, candidates=eligible)
                if picks:
                    card = _card(conn, *picks[0])
            if card is None:
                ids = eligible
                if ids:
                    card = _card(conn, ids[0], None)
            return {"profile": card, "stats": _stats(conn), "model_active": trained is not None}

    @app.get("/api/review")
    def review(model_name: str = "default", limit: int = 100):
        """Audit of the bot's swipes (with your verdict if you gave one) and the cards it left for you."""
        with db.session() as conn:
            trained = model.load(model_name)
            human = db.latest_ratings(conn)

            def entry(pid: int, **kw) -> dict:
                c = _card(conn, pid, model.score_profile(conn, trained, pid) if trained else None)
                return {"id": pid, "name": c["name"], "age": c["age"], "bio": c["bio"], "source": c["source"],
                        "score": c["score"], "photos": c["photos"][:3], **kw}

            auto = []
            for row in db.auto_swipe_log(conn, limit):
                pid = row["profile_id"]
                auto.append(entry(pid, bot_liked=bool(row["liked"]), at=row["created_at"], yours=human.get(pid)))
            pending = []
            if trained:
                for pid, _ in model.most_uncertain(conn, trained, 50, candidates=filters.eligible_unrated(conn)):
                    pending.append(entry(pid))
            unreviewed = [a for a in auto if a["yours"] is None]
            return {"auto": auto, "pending": pending, "model_active": trained is not None,
                    "summary": {"auto": len(auto), "unreviewed": len(unreviewed),
                                "disagreed": sum(a["yours"] is not None and a["yours"] != a["bot_liked"] for a in auto),
                                "agreed": sum(a["yours"] == a["bot_liked"] for a in auto)},
                    "stats": _stats(conn)}

    @app.get("/review")
    def review_page():
        return FileResponse(STATIC / "review.html")

    @app.get("/api/browse")
    def browse(model_name: str = "default"):
        """Read-only list of every saved profile with the model's score and your rating (if any). Rates nothing."""
        with db.session() as conn:
            trained = model.load(model_name)
            human = db.latest_ratings(conn)
            items = []
            for (pid,) in conn.execute("SELECT id FROM profiles ORDER BY id").fetchall():
                p = db.get_profile(conn, pid)
                items.append({"id": pid, "name": p.name, "age": p.age, "source": p.source, "yours": human.get(pid),
                              "score": model.score_profile(conn, trained, pid) if trained else None})
            return {"items": items, "model_active": trained is not None, "sources": sorted({i["source"] for i in items})}

    @app.get("/api/browse/{profile_id}")
    def browse_profile(profile_id: int, model_name: str = "default"):
        """One profile for browsing: card, the model's score, and what drove it. Rates nothing."""
        with db.session() as conn:
            if not conn.execute("SELECT 1 FROM profiles WHERE id=?", (profile_id,)).fetchone():
                raise HTTPException(404, "unknown profile")
            trained = model.load(model_name)
            emb = db.profile_embeddings(conn, profile_id, trained.embed_model) if trained else None
            why = None
            if trained and emb is not None and emb.size:
                why = model.explain_blocks(trained, emb, db.text_embedding(conn, profile_id, trained.embed_model))
            card = _card(conn, profile_id, model.score_profile(conn, trained, profile_id) if trained else None)
            return {"profile": card, "yours": db.latest_ratings(conn).get(profile_id), "why": why,
                    "model_active": trained is not None}

    @app.get("/browse")
    def browse_page():
        return FileResponse(STATIC / "browse.html")

    @app.get("/api/insights")
    def get_insights(source: str | None = None, fake_threshold: int | None = None):
        """Like-rate statistics per attribute (profile data and CLIP visual guesses); see insights.py.

        Profiles with a fake-score (fake.py) >= fake_threshold are excluded; omit it to keep everyone."""
        with db.session() as conn:
            return insights.build(conn, source or None, fake_threshold=fake_threshold)

    @app.get("/insights")
    def insights_page():
        return FileResponse(STATIC / "insights.html")

    @app.get("/api/ideal")
    def get_ideal(source: str | None = None, fake_threshold: int | None = None, model_name: str = "default"):
        """Your preferred value per attribute, an image-generator prompt, and the seen profiles closest to it."""
        with db.session() as conn:
            return ideal.build(conn, source or None, fake_threshold=fake_threshold, model_name=model_name)

    @app.get("/ideal")
    def ideal_page():
        return FileResponse(STATIC / "ideal.html")

    @app.get("/api/filters")
    def get_filters():
        return asdict(filters.load())

    @app.put("/api/filters")
    def put_filters(body: FiltersBody):
        # Fields the client did not send (the panel has no height/field controls yet) keep their saved value.
        f = filters.Filters(**{**asdict(filters.load()), **body.model_dump(exclude_unset=True)})
        if f.min_age is not None and f.max_age is not None and f.min_age > f.max_age:
            raise HTTPException(400, "min_age is greater than max_age")
        filters.save(f)
        with db.session() as conn:
            return {"filters": asdict(f), "stats": _stats(conn)}

    @app.post("/api/rate")
    def rate(body: RateBody):
        with db.session() as conn:
            if db.get_profile(conn, body.profile_id) is None:
                raise HTTPException(404, "no such profile")
            db.add_rating(conn, body.profile_id, body.liked)
            return _stats(conn)

    @app.post("/api/fake")
    def mark_fake(body: FakeBody):
        """Mark a profile fake (also passes it when `and_pass`, so it leaves the rating queue), real, or clear the mark."""
        with db.session() as conn:
            if db.get_profile(conn, body.profile_id) is None:
                raise HTTPException(404, "no such profile")
            db.set_fake_mark(conn, body.profile_id, body.is_fake)
            if body.is_fake and body.and_pass:
                db.add_rating(conn, body.profile_id, False)
            return _stats(conn)

    @app.post("/api/undo")
    def undo():
        with db.session() as conn:
            pid = db.undo_last_rating(conn)
            if pid is not None and db.fake_marks(conn).get(pid) is True:
                db.set_fake_mark(conn, pid, None)  # undoing "fake + pass" restores the profile fully
            return {"undone_profile_id": pid, "stats": _stats(conn)}

    @app.post("/api/profiles")
    async def add_profile(
        photos: list[UploadFile] = File(...),
        name: str | None = Form(None),
        age: int | None = Form(None),
        bio: str | None = Form(None),
        source: str = Form("manual"),
    ):
        with tempfile.TemporaryDirectory() as tmp:
            files = []
            for i, up in enumerate(photos):
                suffix = Path(up.filename or "").suffix.lower()
                if suffix not in ingest.IMAGE_EXTS:
                    raise HTTPException(400, f"unsupported image type: {up.filename}")
                f = Path(tmp) / f"{i}{suffix}"
                f.write_bytes(await up.read())
                files.append(f)
            with db.session() as conn:
                pid = ingest.add_from_files(conn, files, source, external_id=ingest.photos_external_id(files),
                                            name=name or None, age=age, bio=bio or None)
                return {"id": pid, "stats": _stats(conn)}

    @app.post("/api/train")
    def train(model_name: str = "default"):
        from datebot import embed

        with db.session() as conn:
            embedded = embed.embed_pending(conn)
            try:
                t = model.train(conn, embed.EMBED_ID, model_name)
            except ValueError as e:
                raise HTTPException(400, str(e))
            return {"embedded": embedded, "n_samples": t.n_samples, "cv_accuracy": t.cv_accuracy,
                    "stats": _stats(conn)}

    @app.get("/api/eval")
    def evaluation(like_precision: float = 0.9, pass_precision: float = 0.95):
        """Plain-text evaluation report (same as `datebot eval`)."""
        from datebot import embed, evaluate
        from datebot.embed import EMBED_ID

        with db.session() as conn:
            embed.embed_pending(conn)
            try:
                report = evaluate.run(conn, EMBED_ID, like_precision, pass_precision)
            except ValueError as e:
                raise HTTPException(400, str(e))

            def describe(pid: int) -> str:
                p = db.get_profile(conn, pid)
                return f"#{pid} {p.name or ''}".strip()

            return {"text": evaluate.format_report(report, describe), "auc": report.headline.auc, "n": report.n}

    @app.post("/api/reset-model")
    def reset_model(model_name: str = "default"):
        return {"removed": model.reset(model_name)}

    @app.post("/api/reset-ratings")
    def reset_ratings():
        with db.session() as conn:
            return {"deleted": db.clear_ratings(conn), "stats": _stats(conn)}

    return app
