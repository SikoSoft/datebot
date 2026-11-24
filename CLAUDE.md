# CLAUDE.md

## Project

Datebot: learns the user's dating preferences from their own swipes and scores profiles 0–100, to eventually automate swiping on apps like Tinder, Hinge and Badoo. The model must be trainable over time and resettable at will.

## Git workflow (always follow)

- Work directly on `master` unless the user says otherwise.
- Commit every change in clean, separate, logically scoped commits (one concern per commit; never lump unrelated changes together).
- Always push to `origin master` after committing. Never leave commits unpushed.
- Write clear, imperative commit messages.

## Working style

- Ask the user clarifying questions on open decisions rather than guessing.
- Keep all profile data (other people's photos) local by default.

## Setup and commands

- Python 3.12 + `uv`. Always sync with all extras: `uv sync --extra embed --extra ui --extra capture`.
- Tests: `uv run pytest -q -m "not slow"` (slow = CLIP embedding and real-Chrome tests; ~91 fast tests, all passing).
- CLI: `datebot --help`. Main commands: `import`, `embed`, `train`, `score`, `eval`, `ui`, `stats`, `filters`, `reset-model`, `reset-ratings`, `capture tinder|badoo`, `autoswipe tinder`. See `readme.md` for usage.
- All personal data is under gitignored `data/` (override with `DATEBOT_HOME`). Never commit it.

## Architecture (src/datebot)

- `db.py` SQLite. Ratings have an `origin`: `manual` (rated in UI), `app` (your own captured swipe), `auto` (the bot's swipe). **`auto` must never become a training label**; `latest_ratings` already excludes it.
- `embed.py` local CLIP embeddings (photos + bio text). `model.py` features + LogisticRegression, score = P(like)*100, named snapshots in `data/models/`.
- `filters.py` hard deal-breakers (age, distance, keywords) applied before the model.
- `evaluate.py` out-of-fold evaluation and the auto-swipe score bands (`suggest_thresholds`).
- `capture/` passive Chrome listeners: `tinder.py` parsers, `recorder.py`, `browser.py` (listener, observer hooks, PageDriver), `discover.py` (raw traffic dump for unmapped sites).
- `autoswipe/` `logic.py` (bands, card queue, handshake, governor), `runner.py` (controller with halt-on-surprise), `assess.py` (DB/model glue).
- `web/` FastAPI rating UI (`datebot ui`).

## Status (as of 2026-11-24)

Done: rating UI, local model, resettable training, hard filters, text features, Tinder passive capture, evaluation tooling, Tinder auto-swipe (observe-only by default, `--live` to act), Badoo web capture (parser verified on real traffic; `capture/badoo.py`, `badoo_browser.py`; `discover.py` remains for mapping new sites).

**Nothing has been tested against live Tinder or Badoo.** Tinder response field names and the card-order/arrow-key assumptions in auto-swipe are unverified; `capture tinder --dump DIR` saves raw responses to debug.

## Next steps (continue in this order)

1. **Confirm Badoo vote codes with the user** (2=like, 3=pass is assumed; see `capture/badoo.py`) after their first real `capture badoo` session, via `datebot stats`.
2. Possibly Badoo auto-swipe, reusing `autoswipe/` (needs the Badoo card/keys behaviour confirmed first).
3. Hinge is mobile-only: needs phone mirroring or an emulator; ask the user before starting.
4. Have the user collect ~100 real ratings, run `datebot eval`, and tune from the report.
5. Optional: review-queue UI for the cards auto-swipe leaves to the user (currently only announced in the terminal).

## Working rules learned

- Ask the user (AskUserQuestion) before decisions that act on their account or have no documented answer; they want this.
- Automating swipes violates app ToS and risks account bans: keep warnings in the readme, default to observe-only, never add stealth/detection-evasion.
- Don't invent API field names: verify from real traffic, mark unverified parsers clearly.
