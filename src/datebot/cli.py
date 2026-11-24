from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from datebot import __version__, db, filters as hard_filters, ingest, model

app = typer.Typer(help="Learn your dating preferences and score profiles 0-100.", no_args_is_help=True)


@app.callback()
def main() -> None:
    """Datebot."""


@app.command()
def version() -> None:
    """Print the datebot version."""
    typer.echo(__version__)


@app.command("import")
def import_(
    path: Path = typer.Argument(..., exists=True, help="A folder of profile subfolders, or a folder of images."),
    source: str = typer.Option("manual", help="tinder | hinge | badoo | manual"),
) -> None:
    """Import profiles from a folder. Each subfolder is one profile; loose images are one profile each."""
    with db.session() as conn:
        ids = ingest.add_from_folder(conn, path, source)
    typer.echo(f"Imported {len(ids)} profile(s). Next: datebot embed")


@app.command()
def embed() -> None:
    """Compute local CLIP embeddings for any photos that lack them."""
    from datebot import embed as embedder

    with db.session() as conn:
        n = embedder.embed_pending(conn)
    typer.echo(f"Embedded {n} photo(s).")


@app.command()
def train(name: str = typer.Option("default", help="Model name; use different names to keep snapshots.")) -> None:
    """Train a model from your ratings."""
    from datebot.embed import EMBED_ID

    with db.session() as conn:
        try:
            t = model.train(conn, EMBED_ID, name)
        except ValueError as e:
            typer.secho(str(e), fg=typer.colors.RED)
            raise typer.Exit(1)
    cv = f", cross-val accuracy {t.cv_accuracy:.0%}" if t.cv_accuracy is not None else ""
    typer.echo(f"Trained '{name}' on {t.n_samples} ratings ({t.n_likes} likes){cv}.")


@app.command()
def score(
    name: str = typer.Option("default"),
    limit: int = typer.Option(20, help="How many unrated profiles to show."),
    uncertain: bool = typer.Option(False, help="Show the ones the model is least sure about (best to rate next)."),
) -> None:
    """Score unrated profiles 0-100 with a trained model."""
    trained = model.load(name)
    if not trained:
        typer.secho(f"No model '{name}'. Run: datebot train", fg=typer.colors.RED)
        raise typer.Exit(1)
    with db.session() as conn:
        eligible = hard_filters.eligible_unrated(conn)
        if uncertain:
            rows = model.most_uncertain(conn, trained, limit, candidates=eligible)
        else:
            rows = [(pid, model.score_profile(conn, trained, pid)) for pid in eligible]
            rows = sorted((r for r in rows if r[1] is not None), key=lambda r: -r[1])[:limit]
        for pid, s in rows:
            p = db.get_profile(conn, pid)
            typer.echo(f"{s:5.1f}  #{pid}  {p.name or ''}  {db.photo_paths(conn, pid)[0]}")


@app.command()
def models() -> None:
    """List saved models."""
    for name in model.list_models():
        t = model.load(name)
        typer.echo(f"{name}: {t.n_samples} ratings, trained {t.trained_at}" if t
                   else f"{name}: outdated, retrain with: datebot train --name {name}")


@app.command("reset-model")
def reset_model(name: str = typer.Option("default")) -> None:
    """Forget a trained model (ratings are kept, so you can retrain)."""
    typer.echo("Model removed." if model.reset(name) else f"No model '{name}'.")


@app.command("reset-ratings")
def reset_ratings(yes: bool = typer.Option(False, "--yes", help="Skip confirmation.")) -> None:
    """Erase all ratings (start teaching from scratch). Profiles and photos are kept."""
    if not yes:
        typer.confirm("Delete ALL ratings?", abort=True)
    with db.session() as conn:
        n = db.clear_ratings(conn)
    typer.echo(f"Deleted {n} rating(s).")


@app.command()
def stats() -> None:
    """Show counts of profiles and ratings."""
    with db.session() as conn:
        ratings = db.latest_ratings(conn)
        total = conn.execute("SELECT COUNT(*) FROM profiles").fetchone()[0]
    likes = sum(ratings.values())
    typer.echo(f"{total} profiles, {len(ratings)} rated ({likes} likes, {len(ratings) - likes} passes), "
               f"{total - len(ratings)} unrated")


@app.command()
def ui(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Start the local rating UI (needs the `ui` extra: uv sync --extra ui)."""
    import uvicorn

    from datebot.web.app import create_app

    typer.echo(f"Rating UI at http://{host}:{port}")
    uvicorn.run(create_app(), host=host, port=port, log_level="warning")


filters_app = typer.Typer(help="Hard deal-breakers applied before the model.", no_args_is_help=True)
app.add_typer(filters_app, name="filters")


@filters_app.command("show")
def filters_show() -> None:
    """Show the current filters."""
    f = hard_filters.load()
    typer.echo("No filters set." if f.is_empty() else "\n".join(f"{k}: {v}" for k, v in vars(f).items() if v))


@filters_app.command("set")
def filters_set(
    min_age: Optional[int] = typer.Option(None),
    max_age: Optional[int] = typer.Option(None),
    max_distance_km: Optional[float] = typer.Option(None),
    block: Optional[list[str]] = typer.Option(None, help="Reject profiles mentioning this word/phrase (repeatable)."),
    require: Optional[list[str]] = typer.Option(None, help="Reject unless one of these is mentioned (repeatable)."),
) -> None:
    """Update filters. Only the options you pass change; keyword lists are replaced when given."""
    f = hard_filters.load()
    if min_age is not None:
        f.min_age = min_age
    if max_age is not None:
        f.max_age = max_age
    if max_distance_km is not None:
        f.max_distance_km = max_distance_km
    if block:
        f.blocked_keywords = block
    if require:
        f.required_keywords = require
    if f.min_age is not None and f.max_age is not None and f.min_age > f.max_age:
        typer.secho("min-age is greater than max-age", fg=typer.colors.RED)
        raise typer.Exit(1)
    hard_filters.save(f)
    with db.session() as conn:
        hidden = len(db.unrated_profile_ids(conn)) - len(hard_filters.eligible_unrated(conn, f))
    typer.echo(f"Saved. {hidden} unrated profile(s) now hidden.")


@filters_app.command("clear")
def filters_clear() -> None:
    """Remove all filters."""
    hard_filters.save(hard_filters.Filters())
    typer.echo("Filters cleared.")


capture_app = typer.Typer(help="Capture profiles and your swipes from dating apps.", no_args_is_help=True)
app.add_typer(capture_app, name="capture")


@capture_app.command("tinder")
def capture_tinder(
    cdp: Optional[str] = typer.Option(
        None, help="Attach to your own Chrome instead, e.g. http://localhost:9222 "
                   "(start it with --remote-debugging-port=9222 and a separate --user-data-dir)."),
    dump: Optional[Path] = typer.Option(None, help="Save raw Tinder responses here to debug parsing."),
) -> None:
    """Record Tinder profiles and your own swipes while you use tinder.com (needs the `capture` extra).

    You log in and swipe yourself; datebot only listens. Your swipes become labeled ratings.
    """
    from datebot.capture import browser
    from datebot.capture.recorder import Recorder

    rec = Recorder()
    try:
        browser.run(rec, cdp=cdp, dump_dir=dump, log=lambda m: typer.echo(f"[capture] {m}"))
    except KeyboardInterrupt:
        pass
    c = rec.counts
    typer.echo(f"Done: {c.new} new profile(s), {c.swipes} swipe(s) recorded"
               + (f", {c.unknown_swipes} on uncaptured profiles" if c.unknown_swipes else "")
               + (f", {c.failed} skipped (photos failed)" if c.failed else "") + ".")
