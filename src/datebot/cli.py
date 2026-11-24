import typer

from datebot import __version__

app = typer.Typer(help="Learn your dating preferences and score profiles 0-100.", no_args_is_help=True)


@app.command()
def version() -> None:
    """Print the datebot version."""
    typer.echo(__version__)


@app.callback()
def main() -> None:
    """Datebot."""
