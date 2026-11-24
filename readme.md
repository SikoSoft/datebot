# datebot

Learns what you like from your own swipes and scores dating profiles 0-100, with the goal of
eventually automating swiping on Tinder, Hinge and Badoo.

Everything runs locally. Photos, ratings and models live in `data/` (gitignored).

## How it works

1. Profiles (photos + optional name/age/bio) go into a local SQLite database.
2. Each photo is embedded with CLIP (ViT-B-32) on your machine.
3. A logistic-regression model over those embeddings learns your likes/passes and scores profiles 0-100.
4. Once a model exists, the UI shows you the profiles it is *least sure about* first (active learning),
   which is the fastest way to teach it.

## Quick start

```bash
uv sync --extra embed --extra ui
uv run datebot ui            # http://127.0.0.1:8000: drop in profile photos, rate with ←/→, click "Train model"
```

CLI equivalents:

```bash
uv run datebot import ./profiles     # each subfolder = one profile
uv run datebot embed
uv run datebot train
uv run datebot score --uncertain
uv run datebot stats
```

## Resetting

- `datebot reset-model` (or the UI button): forget the trained model; ratings are kept, so you can retrain.
- `datebot reset-ratings`: erase all ratings and teach from scratch.
- `datebot train --name v2`: keep named snapshots side by side.

## Roadmap

- [x] Storage, embeddings, model, CLI, rating UI
- [ ] Bio/prompt text features
- [ ] Hard filters (age, distance, deal-breakers)
- [ ] Capture profiles and swipes from Tinder / Hinge / Badoo
- [ ] Auto-swipe with a confidence band (right above high, left below low, review queue in between)
