# datebot

Learns what you like from your own swipes and scores dating profiles 0-100, with the goal of
eventually automating swiping on Tinder, Hinge and Badoo.

Everything runs locally. Photos, ratings and models live in `data/` (gitignored).

## How it works

1. Profiles (photos + optional name/age/bio) go into a local SQLite database.
2. Each photo, and each bio/prompt text, is embedded with CLIP (ViT-B-32) on your machine.
3. A logistic-regression model over those embeddings learns your likes/passes and scores profiles 0-100.
   Photos of a profile are summarized by their mean and max; the bio is an extra block (zeros if there is none).
4. Hard filters (below) remove deal-breakers before the model ever sees a profile.
5. Once a model exists, the UI shows you the profiles it is *least sure about* first (active learning),
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

## Hard filters

Deal-breakers you already know don't need to be learned. Set them in the UI ("Deal-breakers" panel) or:

```bash
uv run datebot filters set --min-age 25 --max-age 38 --max-distance-km 40 --block smoker --block 420
uv run datebot filters show
uv run datebot filters clear
```

Filtered profiles are hidden from the rating queue and from scoring. A profile missing an age or distance is
never rejected on that basis. Keywords match whole words in the bio and prompts ("smoker" will not match
"non-smoker", but will match "non smoker", so use phrases carefully). `--require` rejects profiles that mention
none of the given words.

## Capturing from Tinder (web)

```bash
uv sync --extra embed --extra ui --extra capture
uv run datebot capture tinder          # opens Chrome with its own saved profile; log in and swipe as normal
uv run datebot embed && uv run datebot train
```

Capture is **passive**: you log in and swipe yourself, and datebot only listens to the responses the page
already receives. Every profile shown is stored with its photos, and each of your likes/passes/super likes
becomes a labeled rating (`origin = app`). Profiles you were shown but have not swiped appear in the rating UI.

- If Tinder rejects the automated Chrome at login, start your own Chrome with
  `--remote-debugging-port=9222 --user-data-dir=/some/dir`, log in there, and run
  `datebot capture tinder --cdp http://localhost:9222` (then reload the Tinder tab).
- Tinder has no public API. Endpoint paths come from community docs, but the response fields are
  best-effort. If parsing finds nothing, run with `--dump ./dump` and check the raw responses.
- Using Tinder with automation tools is against its terms of service; account restrictions are possible.
  Capture only reads; it never swipes for you.

## Capturing from Badoo (web) - discovery step

Badoo's web API is undocumented and not yet mapped, so there is no profile parser yet. This step only records traffic so one can be written from real samples (Bumble, a sibling app, is described in [this write-up](https://medium.com/independent-security-evaluators/reverse-engineering-bumbles-api-a2a0d39b3a87); Badoo may differ).

```bash
uv run datebot capture badoo            # opens a normal Chrome: log in yourself (Google sign-in works), swipe ~15 cards, close the window
```

Saved to `data/badoo-dump/` (override with `--dump`): one JSON file per API call, plus an endpoint summary when you close Chrome. Headers and cookies are never saved, but bodies and URLs may hold session ids and other people's data, so keep the dump private.

## Auto-swipe (Tinder web)

**This acts on your account. Automating swipes breaks Tinder's terms and can get the account restricted or banned.** Use at your own risk, and start without `--live`.

```bash
uv run datebot autoswipe tinder          # observe only: says what it WOULD do, presses nothing
uv run datebot autoswipe tinder --live   # actually presses like/pass when the model is confident
```

- Score cut-offs come from `datebot eval` (or set `--like-at` / `--pass-at`). Confident cards are swiped; everything in between is announced as "YOUR TURN" and left to you.
- Hard filter deal-breakers (age, distance, keywords) are passed automatically.
- Before pressing anything the bot must watch you swipe `--handshake` cards (default 5) exactly where it predicted, proving it knows which card is on top. Its first action is always a pass, to confirm the arrow keys.
- After each press it checks the swipe Tinder received. Wrong profile, wrong direction or no swipe halts it immediately.
- Limits: `--max-swipes` per run (20), `--daily-cap` (60), `--min-delay` seconds between swipes (4), and at least `--min-ratings` of your own ratings (50).
- Its swipes are stored with origin `auto` and are never used as training labels. Every decision is logged to `data/autoswipe.jsonl`.
- Untested against live Tinder: the card-order and arrow-key assumptions are exactly what the safety checks are for. Run observe-only first.

## Is it learning my taste?

```bash
uv run datebot eval          # or the "Evaluate" button in the UI
```

Every number comes from profiles the model was *not* trained on (cross-validation or time-based holdout):

- **ROC AUC** is the headline: 0.5 is a coin flip, 0.7 is decent, 0.8+ is good. Accuracy alone misleads when
  you pass on most profiles, so it is shown next to the "always guess the common answer" baseline.
- **What it learns from:** AUC using photos only, bio text only, and both.
- **Does more rating help?** A learning curve. If it is still rising, keep rating; if flat, more of the same will not help.
- **Has your taste shifted?** Trains on your oldest ratings and tests on your newest.
- **Auto-swipe bands:** the score cut-offs where it is, say, 90% right on likes and 95% right on passes, how many
  profiles that decides, and how many of your real likes an auto-pass band would have lost. Everything between the
  bands is meant for your review queue. These are estimated on the same ratings, so they are optimistic: re-check
  as you rate more. Tune with `--like-precision` and `--pass-precision`.
- **Biggest surprises:** your likes it scored lowest and passes it scored highest. Either blind spots or ratings to reconsider.

With under about 50 ratings the report warns that results are rough.

## Resetting

- `datebot reset-model` (or the UI button): forget the trained model; ratings are kept, so you can retrain.
- `datebot reset-ratings`: erase all ratings and teach from scratch.
- `datebot train --name v2`: keep named snapshots side by side.

## Roadmap

- [x] Storage, embeddings, model, CLI, rating UI
- [x] Bio/prompt text features
- [x] Hard filters (age, distance, deal-breakers)
- [x] Capture profiles and swipes from Tinder web
- [x] Evaluation tooling (`datebot eval`)
- [ ] Capture from Badoo web (discovery step done; parser waits on real traffic samples) and Hinge
- [x] Auto-swipe with a confidence band (Tinder web; the middle band is left to you in the app)
