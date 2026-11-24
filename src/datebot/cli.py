from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from datebot import __version__, db, ingest, model

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
        if uncertain:
            rows = model.most_uncertain(conn, trained, limit)
        else:
            rows = [(pid, model.score_profile(conn, trained, pid)) for pid in db.unrated_profile_ids(conn)]
            rows = sorted((r for r in rows if r[1] is not None), key=lambda r: -r[1])[:limit]
        for pid, s in rows:
            p = db.get_profile(conn, pid)
            typer.echo(f"{s:5.1f}  #{pid}  {p.name or ''}  {db.photo_paths(conn, pid)[0]}")


@app.command()
def models() -> None:
    """List saved models."""
    for name in model.list_models():
        t = model.load(name)
        typer.echo(f"{name}: {t.n_samples} ratings, trained {t.trained_at}")


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
