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


def _embed_new(conn) -> None:
    """Embed profiles that arrived since the last `embed`, so train/score/eval/explain never silently skip them."""
    from datebot import embed as embedder

    try:
        n = embedder.embed_pending(conn)
    except ImportError:
        typer.secho("New photos need embedding but CLIP is not installed. Run: uv sync --extra embed", fg=typer.colors.RED)
        raise typer.Exit(1)
    if n:
        typer.echo(f"Embedded {n} new photo(s) first.")


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
        _embed_new(conn)
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
        _embed_new(conn)
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
        auto = len(db.auto_swiped_ids(conn))
        unrated = len(db.unrated_profile_ids(conn))
        from datebot.embed import EMBED_ID

        pending = db.count_unembedded_profiles(conn, EMBED_ID)
    likes = sum(ratings.values())
    typer.echo(f"{total} profiles, {len(ratings)} rated by you ({likes} likes, {len(ratings) - likes} passes), "
               f"{auto} auto-swiped, {unrated} unrated"
               + (f"; {pending} not embedded yet (train/eval/explain will do it, or run: datebot embed)" if pending else ""))


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
    reject_height_from_cm: Optional[int] = typer.Option(None, help="Reject anyone this tall or taller."),
    field_keyword: Optional[list[str]] = typer.Option(
        None, help="Reject when a labelled field has a word, as FIELD=word (repeatable), e.g. politics=left."),
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
    if reject_height_from_cm is not None:
        f.reject_height_from_cm = reject_height_from_cm
    if field_keyword:
        f.field_keywords = {}
        for item in field_keyword:
            name, sep, word = item.partition("=")
            if not sep or not name.strip() or not word.strip():
                typer.secho(f"--field-keyword needs FIELD=word, got '{item}'", fg=typer.colors.RED)
                raise typer.Exit(1)
            f.field_keywords.setdefault(name.strip().lower(), []).append(word.strip())
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
    keep_first: bool = typer.Option(
        False, "--keep-first",
        help="Also record your first swipe of the session (by default it is ignored: Tinder opens with a "
             "near-universal 'like' that says nothing about your taste)."),
) -> None:
    """Record Tinder profiles and your own swipes while you use tinder.com (needs the `capture` extra).

    You log in and swipe yourself; datebot only listens. Your swipes become labeled ratings, except the
    first swipe of each run, which is ignored unless you pass --keep-first.
    """
    from datebot.capture import browser
    from datebot.capture.recorder import Recorder
    from datebot.interactions import InteractionTracker

    rec = Recorder(skip_first_swipe=not keep_first, tracker=InteractionTracker("tinder", "desktop-chrome"))
    try:
        browser.run(rec, cdp=cdp, dump_dir=dump, log=lambda m: typer.echo(f"[capture] {m}"))
    except KeyboardInterrupt:
        pass
    c = rec.counts
    typer.echo(f"Done: {c.new} new profile(s), {c.swipes} swipe(s) recorded"
               + (f", {c.unknown_swipes} on uncaptured profiles" if c.unknown_swipes else "")
               + (f", {c.failed} skipped (photos failed)" if c.failed else "") + ".")


@capture_app.command("badoo")
def capture_badoo(
    cdp: Optional[str] = typer.Option(None, help="Attach to your own Chrome, e.g. http://localhost:9222."),
    dump: Optional[Path] = typer.Option(None, help="Also save raw Badoo API traffic here (private: it holds session data)."),
) -> None:
    """Record Badoo profiles and your own swipes while you use badoo.com.

    You log in and swipe yourself; datebot only listens. Your swipes become labeled ratings.
    """
    from datebot.capture import badoo_browser
    from datebot.capture.recorder import Recorder
    from datebot.interactions import InteractionTracker

    rec = Recorder(source="badoo", tracker=InteractionTracker("badoo", "desktop-chrome"))
    try:
        badoo_browser.run(rec, cdp=cdp, dump_dir=dump, log=lambda m: typer.echo(f"[badoo] {m}"))
    except KeyboardInterrupt:
        pass
    c = rec.counts
    typer.echo(f"Done: {c.new} new profile(s), {c.swipes} swipe(s) recorded"
               + (f", {c.unknown_swipes} on uncaptured profiles" if c.unknown_swipes else "")
               + (f", {c.failed} skipped (photos failed)" if c.failed else "") + ".")


@capture_app.command("hinge")
def capture_hinge(port: int = typer.Option(8080, help="Port for the proxy your iPhone points at.")) -> None:
    """Record Hinge profiles and your own swipes from your iPhone (needs the `phone` extra).

    Starts a mitmproxy on this Mac; set it up on the phone as in the readme ("Capturing from Hinge"). You use the
    app yourself; datebot only listens. Your likes and passes become labeled ratings. Ctrl+C to stop.
    """
    import subprocess
    import sys

    addon = Path(__file__).parent / "capture" / "hinge_addon.py"
    mitmdump = Path(sys.executable).parent / "mitmdump"
    if not mitmdump.exists():
        raise typer.BadParameter("mitmproxy is not installed: run `uv sync --extra phone`.")
    try:
        subprocess.run([str(mitmdump), "-s", str(addon), "--listen-port", str(port),
                        "--set", "flow_detail=0", "--set", "termlog_verbosity=warn"], check=False)
    except KeyboardInterrupt:
        pass


@capture_app.command("hinge-replay")
def capture_hinge_replay(
    dump: Path = typer.Argument(..., help="A hinge-probe dump folder, e.g. data/hinge-probe."),
    import_: bool = typer.Option(False, "--import", help="Store the profiles and swipes (default: only count them)."),
) -> None:
    """Run a probe dump through the Hinge parser, to check it or to import what was already captured."""
    from datebot.capture import hinge
    from datebot.capture.recorder import Recorder

    session = hinge.HingeSession()
    events = list(hinge.replay(dump, session))
    profiles = [e.profile for e in events if e.kind == "profile"]
    swipes = [e.swipe for e in events if e.kind == "swipe"]
    typer.echo(f"{len({p.external_id for p in profiles})} profile(s), {len(swipes)} swipe(s) "
               f"({sum(s.liked for s in swipes)} likes, {sum(not s.liked for s in swipes)} passes)"
               + (f"; unrecognised ratings {dict(session.unknown_ratings)}" if session.unknown_ratings else ""))
    if not import_:
        typer.echo("Dry run. Add --import to store them.")
        return
    rec = Recorder(source="hinge")
    for e in events:  # in capture order, so a swipe always follows its profile
        rec.record_profiles([e.profile]) if e.kind == "profile" else rec.record_swipe(e.swipe)
    c = rec.counts
    typer.echo(f"Imported {c.new} new profile(s) and {c.swipes} swipe(s)"
               + (f"; {c.unknown_swipes} on uncaptured profiles" if c.unknown_swipes else "")
               + (f"; {c.failed} skipped (photos failed)" if c.failed else "") + ".")


@capture_app.command("hinge-probe")
def capture_hinge_probe() -> None:
    """Summarize what the Hinge mitmproxy probe captured (hosts that are readable vs. certificate-pinned).

    Capture itself runs via mitmdump; see the readme section "Probing Hinge from your iPhone".
    """
    from datebot.capture import hinge_probe

    typer.echo(hinge_probe.summarize())


autoswipe_app = typer.Typer(help="Let the model swipe when it is confident (acts on your account!).", no_args_is_help=True)
app.add_typer(autoswipe_app, name="autoswipe")


def _make_autoswipe(site: str):
    def autoswipe_cmd(
        live: bool = typer.Option(False, "--live", help="Actually press like/pass. Without it the bot only says what it would do."),
        like_at: Optional[float] = typer.Option(None, help="Auto-like at or above this score (default: from `datebot eval`)."),
        pass_at: Optional[float] = typer.Option(None, help="Auto-pass at or below this score (default: from `datebot eval`)."),
        like_precision: float = typer.Option(0.9, min=0.5, max=1.0),
        pass_precision: float = typer.Option(0.95, min=0.5, max=1.0),
        max_swipes: int = typer.Option(20, min=1, help="Stop after this many automatic swipes this run."),
        daily_cap: int = typer.Option(60, min=1, help="Never exceed this many automatic swipes per day."),
        min_delay: float = typer.Option(4.0, min=1.0, help="Minimum seconds between automatic swipes."),
        min_ratings: int = typer.Option(50, min=10, help="Refuse to act with fewer of your own ratings than this."),
        handshake: int = typer.Option(5, min=3, help="Swipes you must make yourself, matching the bot's prediction, before it acts."),
        model_name: str = typer.Option("default", "--model"),
        cdp: Optional[str] = typer.Option(None, help="Attach to your own Chrome, e.g. http://localhost:9222."),
        like_key: str = typer.Option("ArrowRight", help="Key that likes the top card."),
        pass_key: str = typer.Option("ArrowLeft", help="Key that passes the top card."),
    ) -> None:
        """Watch the site and swipe automatically ONLY when the model is confident; leave the rest to you.

        Automating swipes violates the app's terms and can get the account restricted. Start without --live.
        """
        host = {"tinder": "tinder.com", "badoo": "badoo.com"}[site]
        from datebot import embed, filters as hard_filters, model
        from datebot.autoswipe import assess as glue
        from datebot.autoswipe.logic import Bands, Governor, Handshake
        from datebot.autoswipe.runner import AutoSwiper, file_journal
        from datebot.capture import badoo_browser, browser
        from datebot.capture.recorder import Recorder

        trained = model.load(model_name)
        if trained is None:
            typer.echo("No trained model. Rate some profiles and run `datebot train` first.")
            raise typer.Exit(1)
        with db.session() as conn:
            n = len(db.latest_ratings(conn))
            if live and n < min_ratings:
                typer.echo(f"Only {n} ratings by you; need {min_ratings} before acting (--min-ratings to override).")
                raise typer.Exit(1)
            if like_at is None and pass_at is None:
                try:
                    bands = glue.bands_from_eval(conn, like_precision, pass_precision)
                except ValueError as e:
                    typer.echo(f"Cannot pick score bands automatically: {e}")
                    raise typer.Exit(1)
            else:
                bands = Bands(like_at, pass_at)
            today = db.auto_swipes_today(conn)
        if bands.empty:
            typer.echo("No score is trustworthy enough to act on yet (bands are empty). Rate more profiles.")
            raise typer.Exit(1)
        typer.echo(f"Bands: like at >= {bands.like_at}, pass at <= {bands.pass_at}; everything between is yours."
                   f" Mode: {'LIVE' if live else 'observe only'}.")

        rec = Recorder(source=site)
        gov = Governor(max_swipes, daily_cap, already_today=today, min_delay=min_delay)
        from datebot import config
        swiper = AutoSwiper(rec, glue.make_assess(trained, hard_filters.load(), site), bands, gov, Handshake(handshake),
                            driver=None, live=live, like_key=like_key, pass_key=pass_key, log=lambda m: typer.echo(f"[autoswipe] {m}"),
                            journal=file_journal(config.data_dir() / "autoswipe.jsonl"))

        def tick(context) -> bool:
            pages = [p for p in context.pages if host in p.url]
            if pages:
                swiper.driver = browser.PageDriver(pages[0])
            return swiper.step()

        run = browser.run if site == "tinder" else badoo_browser.run
        try:
            run(rec, cdp=cdp, log=lambda m: typer.echo(f"[capture] {m}"), observer=swiper, tick=tick, tick_ms=300)
        except KeyboardInterrupt:
            pass
        typer.echo(swiper.summary())
    return autoswipe_cmd


for _site in ("tinder", "badoo"):
    autoswipe_app.command(_site, help=f"Swipe automatically on {_site} ONLY when the model is confident (acts on your account!).")(_make_autoswipe(_site))


@app.command()
def explain(
    group: Optional[list[str]] = typer.Option(None, help="Only these groups: styling, photo type, body, ethnicity (repeatable)."),
    prompt: Optional[list[str]] = typer.Option(None, help="Extra attribute to test, as a CLIP caption, e.g. 'a photo of a person in a red dress' (repeatable)."),
    examples: int = typer.Option(0, min=0, help="Show this many top-scoring photos per attribute, to check what CLIP finds."),
) -> None:
    """Show which visual attributes (glasses, gym photos, hair, ...) separate your likes from your passes.

    Works after the fact from the stored embeddings: nothing has to be labelled while rating.
    """
    from datebot import embed as embedder, explain as ex
    from datebot.embed import EMBED_ID

    attrs = [a for a in ex.ATTRIBUTES if not group or a.group in group]
    attrs += [ex.Attribute(p, p, "custom") for p in prompt or []]
    if not attrs:
        typer.secho(f"No attributes selected. Groups: {', '.join(ex.GROUPS)}", fg=typer.colors.RED)
        raise typer.Exit(1)
    with db.session() as conn:
        _embed_new(conn)
        pids, paths, emb = db.all_photo_embeddings(conn, EMBED_ID)
        ratings = db.latest_ratings(conn)
    if not len(pids):
        typer.secho("No embedded photos yet. Run: datebot embed", fg=typer.colors.RED)
        raise typer.Exit(1)
    text_emb = embedder.embed_texts([a.prompt for a in attrs])
    try:
        results = ex.analyse(pids, paths, emb, text_emb, attrs, ratings, examples=examples)
    except ValueError as e:
        typer.secho(str(e), fg=typer.colors.RED)
        raise typer.Exit(1)
    likes = sum(1 for pid in set(pids) if ratings.get(pid) is True)
    rated = sum(1 for pid in set(pids) if pid in ratings)
    typer.echo(ex.format_report(results, likes, rated - likes))


@app.command("eval")
def eval_(
    like_precision: float = typer.Option(0.9, min=0.5, max=1.0, help="Target share of auto-liked profiles you really like."),
    pass_precision: float = typer.Option(0.95, min=0.5, max=1.0, help="Target share of auto-passed profiles you really pass."),
) -> None:
    """Report how well the model predicts your taste, what it learns from, and safe auto-swipe thresholds."""
    from datebot import evaluate
    from datebot.embed import EMBED_ID

    with db.session() as conn:
        _embed_new(conn)
        try:
            report = evaluate.run(conn, EMBED_ID, like_precision, pass_precision)
        except ValueError as e:
            typer.secho(str(e), fg=typer.colors.RED)
            raise typer.Exit(1)

        def describe(pid: int) -> str:
            p = db.get_profile(conn, pid)
            return f"#{pid} {p.name or ''} {db.photo_paths(conn, pid)[0]}".replace("  ", " ")

        typer.echo(evaluate.format_report(report, describe))
