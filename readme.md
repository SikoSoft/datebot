# datebot

Learns what you like from your own swipes and scores dating profiles 0-100, with the goal of
eventually automating swiping on Tinder, Hinge and Badoo.

Everything runs locally. Photos, ratings and models live in `data/` (gitignored).

## How it works

1. Profiles (photos + optional name/age/bio) go into a local SQLite database.
2. Each photo, and each bio/prompt text, is embedded with CLIP (ViT-B-32) on your machine.
3. A logistic-regression model over those embeddings learns your likes/passes and scores profiles 0-100.
   Photos of a profile are summarized by their mean, max and first (lead) photo; the bio and each prompt are embedded separately and pooled into an extra block (zeros if there is no text).
4. Hard filters (below) remove deal-breakers before the model ever sees a profile.
5. Once a model exists, the UI shows you the profiles it is *least sure about* first (active learning),
   which is the fastest way to teach it.

## The two steps: `embed` and `train`

Think of datebot as having **two separate jobs** that happen to be chained:

| | embedding (`datebot embed`) | training (`datebot train`) |
|---|---|---|
| **What it does** | *Reads* each photo (and bio) once and converts it into 512 numbers that describe what it looks like. | *Learns your taste*: fits the model to your likes/passes, using those numbers. |
| **Input** | New profiles that have no numbers yet. | Your ratings + the numbers from embedding. |
| **Depends on your ratings?** | No. It never looks at them. | Yes, entirely. |
| **Speed** | The slower one (runs CLIP on your machine; seconds for a few profiles, minutes for hundreds). | Fast (seconds). |
| **Result is permanent?** | Yes. Each photo is embedded once and kept forever (unless you change the CLIP model). | No. It is a snapshot of your ratings *at that moment*. |
| **Needed when** | New profiles arrived (after `import` or `capture`). | Your ratings changed. |

**You rarely need to run `embed` yourself.** `train`, `score`, `eval` and `explain` all embed any new profiles first
("Embedded 12 new photo(s) first.") and then do their job, so the one command to remember is:

```bash
uv run datebot train
```

`datebot embed` still exists if you want to do the slow part ahead of time (for example right after a capture session, so the
next `train` is instant), and `datebot stats` tells you when profiles are waiting ("59 not embedded yet").

Why do `import` and `capture` not embed? They only *store*: capture is a passive listener running next to your browser,
so it is kept light and fast and does not load the CLIP model (and `capture` does not even require the `embed` extra).
Your swipes are saved as ratings the moment you make them, and the profiles' photos are saved to `data/photos`; nothing
is lost by waiting. They just can't be used for learning until embedded, which `train` does for you.

Rules of thumb:

- **Rating a profile does not require any command.** You can rate for an hour; nothing changes until you `train`.
- **Scores do not update by themselves.** The model scores with whatever it learned at the last `train`. The more you have rated
  since, the more out of date it is. Re-train every ~20-30 new ratings, or whenever you want fresh scores.
- **Changing filters needs nothing.** They apply instantly.
- **`autoswipe` embeds** each new profile it sees, but never retrains: it uses the saved model.
- Embedding needs the `embed` extra (`uv sync --extra embed`); if it is missing the commands tell you.

| action | embeds new profiles? | retrains? |
|---|---|---|
| `datebot import`, `datebot capture ...`, rating in the UI | no (they only store) | no |
| `datebot train`, "Train model" button | yes | **yes** |
| `datebot score`, `eval`, `explain`, "Evaluate" button | yes | no |
| `datebot autoswipe` | yes, for each profile it sees | no (uses the saved model) |

## Typical workflows

**First time, or after capturing a batch of profiles**

```bash
uv sync --extra embed --extra ui --extra capture --extra phone
uv run datebot capture tinder         # swipe as usual; your swipes become ratings (stop with Ctrl+C / close Chrome)
                                       # your first swipe each run is ignored (Tinder's free 'perfect 10'); --keep-first records it
uv run datebot train
uv run datebot eval                   # is it learning anything yet?
```

**Rating in the browser**

```bash
uv run datebot ui                     # http://127.0.0.1:8000 : rate with the arrow keys, click "Train model" now and then
```

**Adding profiles from a folder of photos** (each subfolder = one profile)

```bash
uv run datebot import ./profiles
uv run datebot train
```

**"What does it think of profiles I haven't rated?"**

```bash
uv run datebot score --limit 20       # highest scored first
uv run datebot score --uncertain      # the ones it is least sure about: the most useful to rate next
```

**Starting over**

```bash
uv run datebot reset-model            # forget what it learned; ratings and embeddings stay, `train` again any time
uv run datebot reset-ratings          # erase your ratings and teach from scratch
```

## Every command

Run `uv run datebot --help` (or `datebot <command> --help`) for the options. "Needs" is what must have happened first.

### Getting profiles in

| command | what it does | when to run it |
|---|---|---|
| `import PATH [--source manual]` | Copies photos from a folder into the database. Each subfolder is one profile; loose images are one profile each. Stores only; embedding happens at the next `train`. | When you have screenshots/photos to rate by hand. |
| `capture tinder` / `capture badoo` | Opens Chrome; you log in and swipe as usual and datebot records every profile shown plus your own swipes (as ratings). Passive: never swipes for you. Stores only; embedding happens at the next `train`. | Whenever you swipe, to collect data for free. |
| `ui` | Starts the local web app: add profiles, rate them, set filters, train/evaluate/reset with buttons. | Your main way to rate. |

### Teaching the model

| command | what it does | needs | when to run it |
|---|---|---|---|
| `embed` | Converts every not-yet-embedded photo and bio into numbers (CLIP, local). `train`, `score`, `eval` and `explain` do this automatically, so you only need it to do the slow part ahead of time. | profiles imported or captured | Optional. Safe to repeat. |
| `train [--name NAME]` | Fits the model to your likes/passes and saves it (default name `default`; other names keep side-by-side snapshots). Needs at least 3 likes and 3 passes. | rated profiles (new ones are embedded first, automatically) | After rating more, or after capturing/importing profiles you have rated. |
| `score [--limit N] [--uncertain]` | Prints scores 0-100 for profiles you have not rated yet (those passing your filters). | a trained model | To see what it thinks, or to find the best profiles to rate next. |
| `models` | Lists saved models and how many ratings each was trained on. | | To check a snapshot is current. |

### Checking it

| command | what it does | when to run it |
|---|---|---|
| `stats` | Counts: profiles, your likes/passes, bot swipes, unrated, and how many profiles are waiting to be embedded. | Quick sanity check. |
| `eval` | Honest accuracy test on profiles the model was not trained on, plus the auto-swipe score cut-offs. | After training, to learn whether it is worth trusting; re-run as you rate more. |
| `explain [--examples N] [--prompt "..."]` | Which visual attributes (glasses, gym photos, hair, ...) separate your likes from your passes. | After you have a few dozen likes. |

### Controlling it

| command | what it does |
|---|---|
| `filters show / set / clear` | Hard deal-breakers (age, distance, keywords) applied before the model. Take effect instantly. |
| `reset-model [--name NAME]` | Deletes the trained model only. |
| `reset-ratings [--yes]` | Deletes your ratings only (bot swipes are kept). Photos and embeddings are untouched. |

### Acting on your account (read the warnings below first)

| command | what it does |
|---|---|
| `autoswipe tinder` / `autoswipe badoo` | Watches the site and, **only with `--live`**, swipes when the model is confident. Without `--live` it just says what it would do. |
| `version` | Prints the version. |

## Why does it like what it likes?

```bash
uv run datebot explain --examples 3
```

Ranks attributes (glasses, gym photos, hair colour, swimwear, and so on) by how well they separate your likes from your
passes, with no tagging needed: CLIP is asked about each one after the fact. Add your own with `--prompt`.
Needs a few dozen likes before it means much. How embeddings work, where they are stored, and how to analyse
them yourself: [docs/how-it-works.md](docs/how-it-works.md).

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

Rules for things the model can't (and shouldn't) learn from photos:

```bash
uv run datebot filters set --reject-height-from-cm 178 \
  --field-keyword politics=left --field-keyword politics=vänster --block "hjärtat till vänster"
```

- `--reject-height-from-cm N` rejects anyone N cm or taller, read from a "Height: 178 cm" line. Unknown height passes.
- `--field-keyword FIELD=word` matches only inside the labelled line `FIELD: value` (so a bare "left" in a bio
  never triggers, and "Politics: Liberal" passes). Today only profiles whose source captures that field can match.
- Profiles you already rated that trip a rule are **excluded from training and `eval`**, not deleted, so a pass
  caused by a rule never teaches the model what you find unattractive. Re-run `datebot train` after changing rules.

## Capturing from Tinder (web)

```bash
uv sync --extra embed --extra ui --extra capture --extra phone
uv run datebot capture tinder          # opens Chrome with its own saved profile; log in and swipe as normal
uv run datebot train
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

## Capturing from Badoo (web)

```bash
uv run datebot capture badoo            # opens a normal Chrome (logins are remembered); swipe as usual, close it to stop
uv run datebot capture badoo --dump DIR # also save raw API traffic (private: holds session ids and other people's data)
```

Same idea as Tinder: it only listens. Profiles from the encounters feed are stored, and your own swipes become labeled ratings. The wire format was verified against real traffic. Vote codes 2 = like and 3 = pass were confirmed against a real session (they match Bumble's [documented protocol](https://medium.com/independent-security-evaluators/reverse-engineering-bumbles-api-a2a0d39b3a87)). Badoo gives a city but no distance, so the distance filter does not apply. Auto-swipe also works on Badoo, see below.

## Probing Hinge from your iPhone

Hinge is mobile-only, so capture goes through a proxy on this Mac. This first step is a **read-only probe**: it only
records traffic and answers "can we read Hinge's API at all?". Result (Nov 2026): yes, `prod-api.hingeaws.net` is readable
and not pinned; see "Capturing from Hinge" below.

1. `uv tool install mitmproxy`, then `mitmdump -s src/datebot/capture/hinge_probe.py` (listens on port 8080).
2. iPhone, same Wi-Fi: Settings > Wi-Fi > (i) > Configure Proxy > Manual, server = this Mac's LAN IP, port 8080.
3. Safari on the iPhone: open `mitm.it`, install the profile, then Settings > General > About > Certificate Trust
   Settings > enable full trust for the mitmproxy certificate.
4. Open Hinge and browse ~15 profiles. Swipe as you normally would; nothing is sent by datebot.
5. Stop mitmdump, run `datebot capture hinge-probe`. Hosts under "Readable" carry data we can parse. Hosts under
   "refused the proxy certificate" are certificate-pinned: then we fall back to iPhone Mirroring plus screen parsing
   (we will not bypass pinning).
6. Turn the iPhone proxy back off and remove the mitmproxy profile when done: it can read all your phone's traffic.

Dumps live in `data/hinge-probe/` (gitignored): other people's profile data, keep local.

## Capturing from Hinge (iPhone)

Endpoint and data-contract notes for every app live in [docs/api-endpoints.md](docs/api-endpoints.md).

Same phone setup as the probe above (proxy + trusted mitmproxy certificate on the iPhone), but with the real recorder.
Needs the `phone` extra: `uv sync --extra phone`.

1. `datebot capture hinge` (listens on port 8080; `--port` to change). Leave it running.
2. Use Hinge normally on the phone. Profiles (photos, prompts, job/hometown, height) are stored; your likes ("note")
   and passes ("skip") become labeled ratings. Nothing is ever sent by datebot.
3. Ctrl+C to stop; it prints what was recorded. Remember to turn the phone's proxy off afterwards.

`datebot capture hinge-replay data/hinge-probe` runs an earlier probe dump through the same parser (dry run; add
`--import` to store it). A like **without** a comment and roses have not been seen in traffic yet, so they are ignored
and reported as "unrecognised ratings" rather than guessed.

## Behaviour model (how you swipe)

A second model, trained alongside the taste model and independent of it, learns *how* you behave on a profile:
how long you look before swiping, whether you hesitate, scroll or move the pointer, and your pacing over a session.
Every swipe captured by `capture tinder|badoo|hinge` also stores an interaction (linked to the profile and rating when
known; a profile never needs one, so older data stays valid).

What is measured depends on where you swipe:

| source | measured |
|---|---|
| Hinge (iPhone via proxy) | **Timing only**: dwell from the card appearing (previous swipe, or the profile's arrival) to your swipe request. A proxy cannot see touches or scrolls. |
| Tinder / Badoo (desktop Chrome) | Dwell plus input events from a passive injected listener: pointer path, speed, acceleration and deceleration into the click, jitter, pauses, hesitation, wheel up/down and reversals, arrow-key holds, click position inside the button, window focus, and what each click landed on (a descriptor of interface vocabulary: tag, role, aria-label, title, test ids, class tokens, and the text of short buttons only, never bios or names). Clicks are classified as gallery next/previous, open profile, like, pass or other, giving click counts, gallery navigation counts and time-to-open-profile. The keyword rules behind those labels are guesses at Tinder's and Badoo's markup and are unverified against the live sites: check `n_other_clicks` and the stored descriptors, then adjust `behavior.TARGET_RULES` (interactions are re-derived from stored raw events on the next `behavior train`). Scrolling counts mouse-wheel/trackpad events only. **Only arrow-key names are recorded; other keystrokes never are.** The listener is ordinary page JavaScript (it exposes `__datebot_events`), so the page could see it. |

Raw inputs are stored compressed next to the derived features, so new features can be recomputed for old sessions.
Dwell is inferred from timestamps (it includes time spent on anything else between swipes); dwell over 5 minutes counts as a break.

| command | what it does |
|---|---|
| `behavior train [--source hinge] [--edges 0,30,50,70,100] [--name N]` | Groups your swipes by preference score (out-of-fold, so a profile is not scored by a model that memorised your verdict) and prints dwell and input quantiles per band, like/pass dwell, hesitation and session rhythm. Needs 10 scored swipes. Apps differ, so train per `--source` as well as pooled. |
| `behavior show` / `behavior reset` | Print or forget a trained behaviour model. Interactions are kept. |
| `behavior clear-data [--yes]` | Erase all behaviour telemetry. Profiles, photos and ratings are kept. |

The bot's own swipes are stored with origin `auto` and are never learned from. `behavior_model.sample_dwell_ms(model, score)`
draws a dwell from your distribution for that score, but nothing calls it: nothing in datebot uses the behaviour model to act on your account.

## Auto-swipe (Tinder and Badoo web)

**This acts on your account. Automating swipes breaks Tinder's terms and can get the account restricted or banned.** Use at your own risk, and start without `--live`.

```bash
uv run datebot autoswipe tinder          # observe only: says what it WOULD do, presses nothing
uv run datebot autoswipe tinder --live   # actually presses like/pass when the model is confident
uv run datebot autoswipe badoo           # same for badoo.com (observe first!)
```

- Score cut-offs come from `datebot eval` (or set `--like-at` / `--pass-at`). Confident cards are swiped; everything in between is announced as "YOUR TURN" and left to you.
- Hard filter deal-breakers (age, distance, keywords) are passed automatically.
- Before pressing anything the bot must watch you swipe `--handshake` cards (default 5) exactly where it predicted, proving it knows which card is on top. Its first action is always a pass, to confirm the arrow keys.
- After each press it checks the swipe Tinder received. Wrong profile, wrong direction or no swipe halts it immediately.
- Limits: `--max-swipes` per run (20), `--daily-cap` (60), `--min-delay` seconds between swipes (4), and at least `--min-ratings` of your own ratings (50).
- Its swipes are stored with origin `auto` and are never used as training labels. Every decision is logged to `data/autoswipe.jsonl`.
- `/insights` in the UI (`datebot ui`): a wide range of stats about your taste. Each card is one attribute (age, height, distance, lifestyle descriptors, interests, themes in the text, plus CLIP-guessed hair colour, body type, glasses, tattoos, photo content, ...) with your like rate per value against your overall average. Hide any card with the ✕ (remembered in the browser; "Hidden" restores them), filter by app, and sort by strongest signal. Cards marked "guess" are zero-shot CLIP labels: noisy, so check them with `datebot explain --examples`. Uses the stored embeddings; run `datebot embed` first for profiles that lack them.
  - **Fake-profile filter:** `fake.py` gives each profile a 0–100 scam/fake score from transparent signals (no or tiny bio, crypto/investing/Telegram talk, a single photo, @handles, clusters of male-coded interests, studio-polished or heavily filtered photos). The "Hide likely fakes ≥" slider on `/insights` leaves profiles at or above that score out of every stat ("off" keeps all) and lists who was left out, with reasons, so you can tune it. Your own verdict beats the score: 🚩 Fake (key F) in the rating UI marks the current profile fake and passes it; "Actually real" / "Clear my mark" in that list overrides or removes it. It only affects the stats: nothing is deleted, and rating and training are untouched.
- Review what it did at `/review` in the UI (`datebot ui`): the bot's swipes with photo, score and your agree/disagree buttons, plus the cards it left for you. Your verdicts become training ratings, so the model learns where it was wrong.
- Badoo: the arrow keys are only a guess (change with `--like-key` / `--pass-key`). If the site ignores them the bot halts after 6 seconds with "no swipe happened" and has done nothing; that is the safety check working, not a crash. Badoo has no hard distance filter.
- Untested against live Tinder or Badoo: the card-order and arrow-key assumptions are exactly what the safety checks are for. Run observe-only first.

## Is it learning my taste?

```bash
uv run datebot eval          # or the "Evaluate" button in the UI
```

Every number comes from profiles the model was *not* trained on (cross-validation or time-based holdout):

- **ROC AUC** is the headline: 0.5 is a coin flip, 0.7 is decent, 0.8+ is good. Accuracy alone misleads when
  you pass on most profiles, so it is shown next to the "always guess the common answer" baseline.
- **What it learns from:** AUC using photos only, bio text only, and both.
- **By source:** one pooled model learns from every app, so this splits the same cross-validated scores per app
  (Badoo, Tinder, ...) to show where it predicts well and where it does not.
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
- [x] Capture from Badoo web
- [x] Capture from Hinge (iPhone via mitmproxy; auto-swipe on Hinge not built)
- [x] Auto-swipe with a confidence band (Tinder web; the middle band is left to you in the app)
- [x] Review queue UI (audit the bot's swipes, feed back verdicts)
- [x] Behaviour model (dwell, input kinematics and rhythm per score band; collection + distributions, not wired to any action)
